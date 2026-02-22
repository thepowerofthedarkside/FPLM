from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
import re
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from PIL import Image as PILImage
import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Select, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload, selectinload
from starlette.middleware.sessions import SessionMiddleware

from .config import SECRET_KEY
from .database import Base, SessionLocal, engine, get_db
from .deps import get_current_user, require_roles
from .models import (
    Attribute,
    Collection,
    Dictionary,
    DictionaryItem,
    Product,
    ProductAttributeAssignment,
    ProductAttributeValue,
    ProductFile,
    ProductSpec,
    Supplier,
    SystemSetting,
    Task,
    TaskBoard,
    TaskFile,
    TaskQueue,
    User,
)
from .security import hash_password, verify_password
from .services import (
    DATA_TYPES,
    PRODUCT_STATUSES,
    ROLES,
    can_change_attribute_type,
    get_category_items,
    get_value_view,
    validate_and_set_values,
    validate_product_completeness,
)

app = FastAPI(title="FPLM MVP")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
UPLOAD_DIR = Path("app/static/uploads")
PRODUCT_FILES_DIR = Path("app/static/product_files")
TASK_FILES_DIR = Path("app/static/task_files")
FILE_CATEGORIES = ["sketch", "technical_spec", "patterns", "sample_photo", "materials"]
DATA_TYPE_LABELS = {
    "string": "Строка",
    "number": "Число",
    "date": "Дата",
    "bool": "Логический",
    "enum": "Справочник",
}
STATUS_LABELS = {
    "draft": "Черновик",
    "active": "Активен",
    "archived": "Архив",
}
ROLE_LABELS = {
    "admin": "Администратор",
    "user": "Пользователь",
    "guest": "Гость",
}
FILE_CATEGORY_LABELS = {
    "sketch": "Скетч",
    "technical_spec": "Техническое задание",
    "patterns": "Лекала",
    "sample_photo": "Фото образца",
    "materials": "Материалы",
}
SAMPLE_STAGE_LABELS = {
    "proto": "Прототип",
    "salesman_sample": "Презентационный образец",
    "pp_sample": "Предпроизводственный образец",
    "production": "Производство",
}
SAMPLE_STAGES = list(SAMPLE_STAGE_LABELS.keys())
SEASON_LABELS = {"FW": "Осень-Зима", "SS": "Весна-Лето"}
TASK_STATUS_ORDER = ["backlog", "todo", "in_progress", "review", "done"]
TASK_STATUS_LABELS = {
    "backlog": "Бэклог",
    "todo": "К выполнению",
    "in_progress": "В работе",
    "review": "Проверка",
    "done": "Готово",
}
TASK_PRIORITY_LABELS = {
    "low": "Низкий",
    "medium": "Средний",
    "high": "Высокий",
    "critical": "Критический",
}
TASK_PRIORITIES = list(TASK_PRIORITY_LABELS.keys())
SETTINGS_DEFAULTS = {
    "server_base_url": "http://127.0.0.1:8000",
}
SPEC_DICTIONARY_CONFIG: dict[str, dict[str, object]] = {
    "style_type": {
        "code": "product_style_type",
        "name": "Тип изделия",
        "aliases": ["style_type"],
        "items": [("coat", "Пальто"), ("trench", "Тренч"), ("puffer", "Пуховик"), ("jacket", "Куртка")],
    },
    "capsule": {
        "code": "product_capsule",
        "name": "Капсулы",
        "aliases": [],
        "items": [("core", "Core"), ("studio", "Studio"), ("weekend", "Weekend"), ("limited", "Limited")],
    },
    "silhouette": {
        "code": "product_silhouette",
        "name": "Силуэт",
        "aliases": ["silhouette"],
        "items": [("oversize", "Оверсайз"), ("straight", "Прямой"), ("tailored", "Приталенный")],
    },
    "fit_type": {
        "code": "product_fit_type",
        "name": "Посадка",
        "aliases": ["fit_type", "t_type"],
        "items": [("regular", "Regular"), ("relaxed", "Relaxed"), ("slim", "Slim")],
    },
    "shell_material": {
        "code": "product_shell_material",
        "name": "Материал верха",
        "aliases": [],
        "items": [("wool", "Шерсть"), ("polyamide", "Полиамид"), ("membrane", "Мембрана")],
    },
    "lining_material": {
        "code": "product_lining_material",
        "name": "Материал подкладки",
        "aliases": ["lining_material"],
        "items": [("viscose", "Вискоза"), ("polyester", "Полиэстер"), ("cotton", "Хлопок")],
    },
    "insulation": {
        "code": "product_insulation",
        "name": "Утеплитель",
        "aliases": ["insulation"],
        "items": [("down", "Пух"), ("synthetic", "Синтетический"), ("none", "Без утеплителя")],
    },
}


def _next_auto_code(codes: list[str], prefix: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    max_num = 0
    for code in codes:
        match = pattern.match((code or "").strip().upper())
        if not match:
            continue
        try:
            max_num = max(max_num, int(match.group(1)))
        except ValueError:
            continue
    return f"{prefix}-{max_num + 1:02d}"


def _apply_task_filters(
    stmt: Select[tuple[Task]],
    q: str,
    status: str,
    priority: str,
    queue_id: int | None,
    assignee_id: int | None,
    collection_id: int | None,
    product_id: int | None,
) -> Select[tuple[Task]]:
    if q:
        like = f"%{q}%"
        stmt = stmt.outerjoin(User, User.id == Task.assignee_id).outerjoin(Product, Product.id == Task.product_id).outerjoin(
            Collection, Collection.id == Task.collection_id
        ).where(
            or_(
                Task.title.ilike(like),
                Task.comment.ilike(like),
                Task.tags.ilike(like),
                Task.status.ilike(like),
                Task.priority.ilike(like),
                User.login.ilike(like),
                Product.sku.ilike(like),
                Product.name.ilike(like),
                Collection.code.ilike(like),
                Collection.name.ilike(like),
            )
        )
    if status and status in TASK_STATUS_ORDER:
        stmt = stmt.where(Task.status == status)
    if priority and priority in TASK_PRIORITIES:
        stmt = stmt.where(Task.priority == priority)
    if queue_id:
        stmt = stmt.where(Task.queue_id == queue_id)
    if assignee_id:
        stmt = stmt.where(Task.assignee_id == assignee_id)
    if collection_id:
        stmt = stmt.where(Task.collection_id == collection_id)
    if product_id:
        stmt = stmt.where(Task.product_id == product_id)
    return stmt


@app.middleware("http")
async def force_utf8_html(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html"):
        response.headers["content-type"] = "text/html; charset=utf-8"
    return response


def _set_flash(request: Request, message: str, level: str = "info") -> None:
    request.session["flash"] = {"message": message, "level": level}


def _get_flash(request: Request) -> dict[str, str] | None:
    return request.session.pop("flash", None)


def _build_breadcrumbs(path: str, title: str | None) -> list[dict[str, str | None]]:
    current_title = (title or "").strip() or "Страница"
    items: list[dict[str, str | None]] = []

    if path == "/login":
        return [{"label": "Р’С…РѕРґ", "url": None}]
    if path == "/cabinet":
        return [{"label": "Кабинет", "url": None}]
    if path == "/settings":
        return [{"label": "Настройки", "url": None}]

    if path == "/products":
        return [{"label": "Изделия", "url": None}]
    if re.match(r"^/products/\d+$", path):
        return [{"label": "Изделия", "url": "/products"}, {"label": current_title, "url": None}]
    if re.match(r"^/tasks/by-product/\d+$", path):
        return [{"label": "Изделия", "url": "/products"}, {"label": current_title, "url": None}]
    m = re.match(r"^/products/(\d+)/attributes$", path)
    if m:
        return [{"label": "Изделия", "url": "/products"}, {"label": "Атрибуты изделия", "url": None}]
    m = re.match(r"^/products/(\d+)/attributes/\d+$", path)
    if m:
        product_id = m.group(1)
        return [
            {"label": "Изделия", "url": "/products"},
            {"label": "Атрибуты изделия", "url": f"/products/{product_id}/attributes"},
            {"label": current_title, "url": None},
        ]

    if path == "/queues":
        return [{"label": "Очереди задач", "url": None}]
    if re.match(r"^/queues/\d+$", path):
        return [{"label": "Очереди задач", "url": "/queues"}, {"label": current_title, "url": None}]

    if path == "/boards":
        return [{"label": "Канбан", "url": None}]
    m = re.match(r"^/boards/(\d+)$", path)
    if m:
        return [{"label": "Канбан", "url": "/boards"}, {"label": current_title, "url": None}]
    m = re.match(r"^/boards/(\d+)/kanban$", path)
    if m:
        board_id = m.group(1)
        return [
            {"label": "Канбан", "url": "/boards"},
            {"label": "Доска", "url": f"/boards/{board_id}"},
            {"label": current_title, "url": None},
        ]

    if re.match(r"^/tasks/\d+$", path):
        return [{"label": "Очереди задач", "url": "/queues"}, {"label": current_title, "url": None}]

    if path == "/collections":
        return [{"label": "Коллекции", "url": None}]
    if re.match(r"^/collections/\d+$", path):
        return [{"label": "Коллекции", "url": "/collections"}, {"label": current_title, "url": None}]

    if path == "/suppliers":
        return [{"label": "Поставщики", "url": None}]
    if re.match(r"^/suppliers/\d+$", path):
        return [{"label": "Поставщики", "url": "/suppliers"}, {"label": current_title, "url": None}]

    if path == "/attributes":
        return [{"label": "Атрибуты", "url": None}]
    if re.match(r"^/attributes/\d+$", path):
        return [{"label": "Атрибуты", "url": "/attributes"}, {"label": current_title, "url": None}]

    if path == "/dictionaries":
        return [{"label": "Справочники", "url": None}]
    if re.match(r"^/dictionaries/\d+$", path):
        return [{"label": "Справочники", "url": "/dictionaries"}, {"label": current_title, "url": None}]

    if path == "/users":
        return [{"label": "Пользователи", "url": None}]
    if re.match(r"^/users/\d+$", path):
        return [{"label": "Пользователи", "url": "/users"}, {"label": current_title, "url": None}]

    items.append({"label": current_title, "url": None})
    return items


def _render(request: Request, template_name: str, context: dict) -> HTMLResponse:
    context["request"] = request
    context["flash"] = _get_flash(request)
    breadcrumbs = _build_breadcrumbs(request.url.path, context.get("title"))
    context["breadcrumbs"] = breadcrumbs if len(breadcrumbs) > 1 else []
    context.setdefault("data_type_labels", DATA_TYPE_LABELS)
    context.setdefault("status_labels", STATUS_LABELS)
    context.setdefault("role_labels", ROLE_LABELS)
    context.setdefault("file_category_labels", FILE_CATEGORY_LABELS)
    context.setdefault("sample_stage_labels", SAMPLE_STAGE_LABELS)
    context.setdefault("season_labels", SEASON_LABELS)
    context.setdefault("task_status_labels", TASK_STATUS_LABELS)
    context.setdefault("task_priority_labels", TASK_PRIORITY_LABELS)
    if "user" not in context:
        user_id = request.session.get("user_id")
        if user_id:
            db = SessionLocal()
            try:
                context["user"] = db.get(User, user_id)
            finally:
                db.close()
        else:
            context["user"] = None
    return templates.TemplateResponse(template_name, context)


def _redirect(url: str, status_code: int = 303) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=status_code)


def _get_setting(db: Session, key: str, default: str = "") -> str:
    row = db.scalar(select(SystemSetting).where(SystemSetting.key == key))
    return row.value if row else default


def _get_server_base_url() -> str:
    db = SessionLocal()
    try:
        raw = _get_setting(db, "server_base_url", SETTINGS_DEFAULTS["server_base_url"]).strip()
        return raw.rstrip("/")
    finally:
        db.close()


def _ensure_uploads_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCT_FILES_DIR.mkdir(parents=True, exist_ok=True)
    TASK_FILES_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_product_columns() -> None:
    # Lightweight SQLite migration for existing DBs.
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(products)").fetchall()
        columns = {row[1] for row in rows}
        if "cover_image_path" not in columns:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN cover_image_path VARCHAR(300)")
        if "designer_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN designer_id INTEGER")
        if "product_manager_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN product_manager_id INTEGER")
        if "pattern_maker_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN pattern_maker_id INTEGER")
        if "technologist_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN technologist_id INTEGER")
        if "department_head_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN department_head_id INTEGER")


def _ensure_user_columns() -> None:
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
        columns = {row[1] for row in rows}
        if "full_name" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN full_name VARCHAR(200)")
        if "department" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN department VARCHAR(200)")
        if "position" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN position VARCHAR(200)")
        if "department_item_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN department_item_id INTEGER")
        if "position_item_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN position_item_id INTEGER")


def _ensure_board_columns() -> None:
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(task_boards)").fetchall()
        columns = {row[1] for row in rows}
        if "filter_queue_id" not in columns:
            conn.exec_driver_sql("ALTER TABLE task_boards ADD COLUMN filter_queue_id INTEGER")


def _ensure_product_spec_columns() -> None:
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(product_specs)").fetchall()
        columns = {row[1] for row in rows}
        if "capsule" not in columns:
            conn.exec_driver_sql("ALTER TABLE product_specs ADD COLUMN capsule VARCHAR(120)")


def _get_dictionary_items(db: Session, code: str) -> tuple[Dictionary | None, list[DictionaryItem]]:
    dictionary = db.scalar(select(Dictionary).where(Dictionary.code == code))
    if not dictionary:
        return None, []
    items = list(
        db.scalars(
            select(DictionaryItem)
            .where(DictionaryItem.dictionary_id == dictionary.id, DictionaryItem.is_active.is_(True))
            .order_by(DictionaryItem.sort_order.asc(), DictionaryItem.label.asc())
        ).all()
    )
    return dictionary, items


def _merge_dictionary_records(db: Session, source: Dictionary, target: Dictionary) -> None:
    if source.id == target.id:
        return

    target_items = list(
        db.scalars(select(DictionaryItem).where(DictionaryItem.dictionary_id == target.id).order_by(DictionaryItem.id.asc())).all()
    )
    by_code = {item.code: item for item in target_items}
    by_label = {item.label: item for item in target_items}
    next_sort = max([item.sort_order for item in target_items] + [0]) + 1

    source_items = list(
        db.scalars(select(DictionaryItem).where(DictionaryItem.dictionary_id == source.id).order_by(DictionaryItem.id.asc())).all()
    )
    for src_item in source_items:
        mapped_item = by_code.get(src_item.code) or by_label.get(src_item.label)
        if not mapped_item:
            # Move source item into canonical dictionary if no equivalent exists.
            src_item.dictionary_id = target.id
            src_item.sort_order = max(src_item.sort_order, next_sort)
            next_sort = src_item.sort_order + 1
            by_code[src_item.code] = src_item
            by_label[src_item.label] = src_item
            mapped_item = src_item

        if mapped_item.id != src_item.id:
            db.query(ProductAttributeValue).filter(ProductAttributeValue.dictionary_item_id == src_item.id).update(
                {ProductAttributeValue.dictionary_item_id: mapped_item.id},
                synchronize_session=False,
            )
            db.query(Product).filter(Product.category_item_id == src_item.id).update(
                {Product.category_item_id: mapped_item.id},
                synchronize_session=False,
            )
            db.query(User).filter(User.department_item_id == src_item.id).update(
                {User.department_item_id: mapped_item.id},
                synchronize_session=False,
            )
            db.query(User).filter(User.position_item_id == src_item.id).update(
                {User.position_item_id: mapped_item.id},
                synchronize_session=False,
            )
            db.delete(src_item)

    db.query(Attribute).filter(Attribute.dictionary_id == source.id).update(
        {Attribute.dictionary_id: target.id},
        synchronize_session=False,
    )
    db.delete(source)


def _normalize_spec_dictionaries(db: Session) -> None:
    for field_name, cfg in SPEC_DICTIONARY_CONFIG.items():
        canonical_code = str(cfg["code"])
        canonical_name = str(cfg["name"])
        aliases = [str(x) for x in cfg.get("aliases", [])]

        canonical = db.scalar(select(Dictionary).where(Dictionary.code == canonical_code))
        alias_dicts = [db.scalar(select(Dictionary).where(Dictionary.code == alias)) for alias in aliases]
        alias_dicts = [d for d in alias_dicts if d is not None]

        if not canonical and alias_dicts:
            # Reuse existing legacy dictionary as canonical, avoiding a duplicate create.
            canonical = alias_dicts[0]
            canonical.code = canonical_code
            canonical.name = canonical_name
            canonical.description = f"РЎРїСЂР°РІРѕС‡РЅРёРє РґР»СЏ РїРѕР»СЏ '{canonical_name}'"
            alias_dicts = alias_dicts[1:]

        if not canonical:
            continue

        for alias_dict in alias_dicts:
            if alias_dict.id == canonical.id:
                continue
            _merge_dictionary_records(db, alias_dict, canonical)


def _resolve_spec_dictionary(db: Session, field_name: str) -> Dictionary | None:
    cfg = SPEC_DICTIONARY_CONFIG.get(field_name)
    if not cfg:
        return None
    code = str(cfg["code"])
    aliases = [str(x) for x in cfg.get("aliases", [])]
    name = str(cfg["name"])

    dictionary = db.scalar(select(Dictionary).where(Dictionary.code == code))
    if dictionary:
        return dictionary

    for alias in aliases:
        dictionary = db.scalar(select(Dictionary).where(Dictionary.code == alias))
        if dictionary:
            return dictionary

    dictionary = db.scalar(select(Dictionary).where(Dictionary.name == name))
    return dictionary


def _get_spec_dictionary_options(db: Session) -> dict[str, list[DictionaryItem]]:
    options: dict[str, list[DictionaryItem]] = {}
    for field_name in SPEC_DICTIONARY_CONFIG:
        dictionary = _resolve_spec_dictionary(db, field_name)
        if not dictionary:
            options[field_name] = []
            continue
        items = list(
            db.scalars(
                select(DictionaryItem)
                .where(DictionaryItem.dictionary_id == dictionary.id, DictionaryItem.is_active.is_(True))
                .order_by(DictionaryItem.sort_order.asc(), DictionaryItem.label.asc())
            ).all()
        )
        options[field_name] = items
    return options


def _dictionary_item_usage_counts(db: Session, item_id: int) -> tuple[int, int]:
    used_as_category = int(db.scalar(select(func.count(Product.id)).where(Product.category_item_id == item_id)) or 0)
    used_in_attribute_values = int(
        db.scalar(select(func.count(ProductAttributeValue.id)).where(ProductAttributeValue.dictionary_item_id == item_id)) or 0
    )
    return used_as_category, used_in_attribute_values


def _dictionary_usage_counts(db: Session, dictionary_id: int) -> tuple[int, int, int]:
    used_by_attributes = int(db.scalar(select(func.count(Attribute.id)).where(Attribute.dictionary_id == dictionary_id)) or 0)
    used_in_categories = int(
        db.scalar(
            select(func.count(Product.id))
            .join(DictionaryItem, DictionaryItem.id == Product.category_item_id)
            .where(DictionaryItem.dictionary_id == dictionary_id)
        )
        or 0
    )
    used_in_attribute_values = int(
        db.scalar(
            select(func.count(ProductAttributeValue.id))
            .join(DictionaryItem, DictionaryItem.id == ProductAttributeValue.dictionary_item_id)
            .where(DictionaryItem.dictionary_id == dictionary_id)
        )
        or 0
    )
    return used_by_attributes, used_in_categories, used_in_attribute_values


def _next_available_sort_order(db: Session, dictionary_id: int) -> int:
    used_orders = sorted(
        {
            int(v)
            for v in db.scalars(
                select(DictionaryItem.sort_order).where(DictionaryItem.dictionary_id == dictionary_id)
            ).all()
            if v is not None and int(v) > 0
        }
    )
    next_value = 1
    for value in used_orders:
        if value != next_value:
            break
        next_value += 1
    return next_value


def _is_valid_category_item(db: Session, category_item_id: int) -> bool:
    item = db.scalar(
        select(DictionaryItem)
        .join(Dictionary, Dictionary.id == DictionaryItem.dictionary_id)
        .where(
            DictionaryItem.id == category_item_id,
            DictionaryItem.is_active.is_(True),
            Dictionary.code == "category",
        )
    )
    return item is not None


def _collection_usage_count(db: Session, collection_id: int) -> int:
    return int(db.scalar(select(func.count(ProductSpec.id)).where(ProductSpec.collection_id == collection_id)) or 0)


def _supplier_usage_count(db: Session, supplier_id: int) -> int:
    return int(db.scalar(select(func.count(ProductSpec.id)).where(ProductSpec.supplier_id == supplier_id)) or 0)


def _queue_usage_count(db: Session, queue_id: int) -> int:
    return int(db.scalar(select(func.count(Task.id)).where(Task.queue_id == queue_id)) or 0)


def _queue_board_usage_count(db: Session, queue_id: int) -> int:
    return int(db.scalar(select(func.count(TaskBoard.id)).where(TaskBoard.filter_queue_id == queue_id)) or 0)


def _board_usage_count(db: Session, board_id: int) -> int:
    return int(db.scalar(select(func.count(Task.id)).where(Task.board_id == board_id)) or 0)


def _is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value))


def _user_usage_counts(db: Session, user_id: int) -> dict[str, int]:
    return {
        "products_team": int(
            db.scalar(
                select(func.count(Product.id)).where(
                    or_(
                        Product.designer_id == user_id,
                        Product.product_manager_id == user_id,
                        Product.pattern_maker_id == user_id,
                        Product.technologist_id == user_id,
                        Product.department_head_id == user_id,
                    )
                )
            )
            or 0
        ),
        "tasks_owner": int(
            db.scalar(
                select(func.count(Task.id)).where(or_(Task.author_id == user_id, Task.assignee_id == user_id))
            )
            or 0
        ),
    }


def _ensure_attribute_assigned_to_all_products(db: Session, attribute_id: int) -> int:
    product_ids = list(db.scalars(select(Product.id)).all())
    if not product_ids:
        return 0
    assigned_product_ids = set(
        db.scalars(
            select(ProductAttributeAssignment.product_id).where(ProductAttributeAssignment.attribute_id == attribute_id)
        ).all()
    )
    missing_product_ids = [product_id for product_id in product_ids if product_id not in assigned_product_ids]
    for product_id in missing_product_ids:
        db.add(ProductAttributeAssignment(product_id=product_id, attribute_id=attribute_id))
    return len(missing_product_ids)


def _ensure_product_has_all_active_attributes(db: Session, product_id: int) -> bool:
    active_attribute_ids = list(db.scalars(select(Attribute.id).where(Attribute.is_active.is_(True))).all())
    if not active_attribute_ids:
        return False
    assigned_attribute_ids = set(
        db.scalars(
            select(ProductAttributeAssignment.attribute_id).where(ProductAttributeAssignment.product_id == product_id)
        ).all()
    )
    missing_attribute_ids = [attr_id for attr_id in active_attribute_ids if attr_id not in assigned_attribute_ids]
    for attr_id in missing_attribute_ids:
        db.add(ProductAttributeAssignment(product_id=product_id, attribute_id=attr_id))
    return bool(missing_attribute_ids)


def _load_product_attribute_assignments(db: Session, product_id: int) -> list[ProductAttributeAssignment]:
    changed = _ensure_product_has_all_active_attributes(db, product_id)
    if changed:
        db.commit()
    return list(
        db.scalars(
            select(ProductAttributeAssignment)
            .options(
                joinedload(ProductAttributeAssignment.attribute),
                selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
            )
            .join(ProductAttributeAssignment.attribute)
            .where(
                ProductAttributeAssignment.product_id == product_id,
                Attribute.is_active.is_(True),
            )
            .order_by(Attribute.name.asc(), ProductAttributeAssignment.id.asc())
        ).all()
    )


def _product_matches_full_search(
    product: Product,
    q_norm: str,
    assignments: list[ProductAttributeAssignment],
) -> bool:
    haystack: list[str] = [
        product.sku or "",
        product.name or "",
        product.description or "",
        product.status or "",
    ]
    if product.category_item:
        haystack.extend([product.category_item.code or "", product.category_item.label or ""])
    if product.spec:
        spec = product.spec
        haystack.extend(
            [
                spec.style_type or "",
                spec.capsule or "",
                spec.silhouette or "",
                spec.fit_type or "",
                spec.shell_material or "",
                spec.lining_material or "",
                spec.insulation or "",
                spec.sample_stage or "",
                str(spec.length_cm) if spec.length_cm is not None else "",
                str(spec.planned_cost) if spec.planned_cost is not None else "",
                str(spec.actual_cost) if spec.actual_cost is not None else "",
            ]
        )
        if spec.collection:
            haystack.extend([spec.collection.code or "", spec.collection.name or ""])
        if spec.supplier:
            haystack.extend([spec.supplier.code or "", spec.supplier.name or "", spec.supplier.country or ""])
    for assignment in assignments:
        if not assignment.attribute:
            continue
        haystack.append(assignment.attribute.name or "")
        if assignment.values:
            haystack.extend(get_value_view(v, assignment.attribute.data_type) for v in assignment.values)
    return q_norm in " ".join(haystack).casefold()


def _product_matches_attribute_filters(
    product: Product,
    assignments: list[ProductAttributeAssignment],
    attributes: list[Attribute],
    filter_values: dict[int, dict[str, object]],
) -> bool:
    assignments_by_attr = {a.attribute_id: a for a in assignments}
    for attr in attributes:
        cfg = filter_values.get(attr.id, {})
        mode = str(cfg.get("mode", ""))
        if mode == "bool":
            raw = str(cfg.get("value", "")).strip()
            if raw == "":
                continue
            expected = raw == "1"
            assignment = assignments_by_attr.get(attr.id)
            if not assignment or not assignment.values:
                return False
            actual_values = [bool(v.value_bool) for v in assignment.values if v.value_bool is not None]
            if not actual_values or expected not in actual_values:
                return False
            continue

        if mode == "enum":
            selected_ids = [int(v) for v in cfg.get("values", []) if str(v).strip().isdigit()]
            if not selected_ids:
                continue
            assignment = assignments_by_attr.get(attr.id)
            if not assignment or not assignment.values:
                return False
            assigned_ids = {
                int(v.dictionary_item_id)
                for v in assignment.values
                if v.dictionary_item_id is not None
            }
            if not assigned_ids.intersection(selected_ids):
                return False
            continue

        needle = str(cfg.get("value", "")).strip().casefold()
        if not needle:
            continue
        assignment = assignments_by_attr.get(attr.id)
        if not assignment or not assignment.values:
            return False
        value_texts = [get_value_view(v, attr.data_type).casefold() for v in assignment.values]
        if not any(needle in value_text for value_text in value_texts):
            return False

    return True


def _product_collection_map(db: Session) -> dict[int, int | None]:
    rows = db.execute(
        select(Product.id, ProductSpec.collection_id).outerjoin(ProductSpec, ProductSpec.product_id == Product.id)
    ).all()
    return {int(product_id): (int(collection_id) if collection_id is not None else None) for product_id, collection_id in rows}


def _resolve_task_collection_id(db: Session, product_id: int | None, fallback_collection_id: int | None) -> int | None:
    if product_id is None:
        return fallback_collection_id
    return db.scalar(select(ProductSpec.collection_id).where(ProductSpec.product_id == product_id))


def _cover_palette(style_type: str | None) -> tuple[str, str, str]:
    key = (style_type or "").lower()
    if "trench" in key:
        return ("#d4b58f", "#f4e5cf", "TR")
    if "puffer" in key:
        return ("#9aa4c1", "#e1e6f2", "PF")
    if "coat" in key:
        return ("#9b7f74", "#f0e0db", "CT")
    if "jacket" in key:
        return ("#6d7f8f", "#dae4ec", "JK")
    return ("#7d8a9a", "#e5ebf1", "PR")


@app.get("/product-cover/{product_id}.svg")
def product_cover(product_id: int, db: Session = Depends(get_db)) -> Response:
    product = db.scalar(select(Product).options(selectinload(Product.spec)).where(Product.id == product_id))
    if not product:
        return Response(content="", status_code=404)
    if product.cover_image_path:
        return _redirect(product.cover_image_path, status_code=307)

    accent, bg, token = _cover_palette(product.spec.style_type if product.spec else None)
    title = (product.name or "Product")[:22]
    sku = (product.sku or "")[:16]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="460" height="280" viewBox="0 0 460 280">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="#ffffff"/>
    </linearGradient>
  </defs>
  <rect width="460" height="280" rx="22" fill="url(#g)"/>
  <circle cx="392" cy="58" r="30" fill="{accent}" opacity="0.25"/>
  <rect x="24" y="24" width="72" height="38" rx="10" fill="{accent}"/>
  <text x="60" y="49" text-anchor="middle" font-family="Segoe UI, Arial" font-size="18" font-weight="700" fill="#fff">{token}</text>
  <text x="24" y="118" font-family="Segoe UI, Arial" font-size="30" font-weight="700" fill="#1f2937">{sku}</text>
  <text x="24" y="156" font-family="Segoe UI, Arial" font-size="22" fill="#334155">{title}</text>
  <rect x="24" y="200" width="180" height="12" rx="6" fill="#cbd5e1"/>
  <rect x="24" y="220" width="140" height="12" rx="6" fill="#e2e8f0"/>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


def _qr_png(payload: str) -> bytes:
    img = qrcode.make(payload)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _register_pdf_font() -> str:
    font_name = "Helvetica"
    font_paths = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in font_paths:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("FPLMUnicode", str(font_path)))
            font_name = "FPLMUnicode"
            break
        except Exception:
            continue
    return font_name


def _product_pdf_bytes(
    product: Product,
    attribute_assignments: list[ProductAttributeAssignment],
    tasks: list[Task],
) -> bytes:
    font_name = _register_pdf_font()
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    normal.fontName = font_name
    normal.fontSize = 9
    title_style = styles["Heading1"]
    title_style.fontName = font_name
    title_style.fontSize = 15
    heading_style = styles["Heading3"]
    heading_style.fontName = font_name
    heading_style.fontSize = 11

    def p(text: str) -> Paragraph:
        return Paragraph(escape(text), normal)

    flow: list = []

    def cover_for_pdf() -> RLImage | Paragraph:
        if not product.cover_image_path or not product.cover_image_path.startswith("/static/"):
            return p("Обложка не загружена")
        local_path = Path("app") / product.cover_image_path.lstrip("/")
        if not local_path.exists() or not local_path.is_file():
            return p("Обложка не найдена")
        try:
            max_w_mm = 86
            max_h_mm = 56
            with PILImage.open(local_path) as img:
                rgb = img.convert("RGB")
                src_w, src_h = rgb.size
                buf = BytesIO()
                rgb.save(buf, format="PNG")
                buf.seek(0)
            if src_w <= 0 or src_h <= 0:
                return p("Обложка недоступна")
            scale = min((max_w_mm * mm) / src_w, (max_h_mm * mm) / src_h)
            draw_w = max(1, src_w * scale)
            draw_h = max(1, src_h * scale)
            cover = RLImage(buf, width=draw_w, height=draw_h)
            cover.hAlign = "RIGHT"
            return cover
        except Exception:
            return p("Обложка недоступна")

    flow.append(Paragraph(f"Карточка изделия: {escape(product.sku)}", title_style))
    flow.append(Spacer(1, 4 * mm))

    category_label = product.category_item.label if product.category_item else "-"
    details_table = Table(
        [
            ["SKU", product.sku or ""],
            ["Наименование", product.name or ""],
            ["Статус", STATUS_LABELS.get(product.status, product.status)],
            ["Категория", category_label],
            ["Описание", product.description or "-"],
        ],
        colWidths=[30 * mm, 58 * mm],
    )
    details_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    top_block = Table([[details_table, cover_for_pdf()]], colWidths=[93 * mm, 93 * mm])
    top_block.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    flow.append(top_block)

    flow.append(Spacer(1, 5 * mm))
    flow.append(Paragraph("Fashion-спецификация", heading_style))
    spec = product.spec
    spec_rows = [
        ["Коллекция", f"{spec.collection.code} - {spec.collection.name}" if spec and spec.collection else "-"],
        ["Поставщик", spec.supplier.name if spec and spec.supplier else "-"],
        ["Тип изделия", spec.style_type if spec and spec.style_type else "-"],
        ["Капсула", spec.capsule if spec and spec.capsule else "-"],
        ["Этап образца", SAMPLE_STAGE_LABELS.get(spec.sample_stage, spec.sample_stage) if spec and spec.sample_stage else "-"],
        ["Силуэт", spec.silhouette if spec and spec.silhouette else "-"],
        ["Посадка", spec.fit_type if spec and spec.fit_type else "-"],
        ["Длина (см)", str(spec.length_cm) if spec and spec.length_cm is not None else "-"],
        ["Материал верха", spec.shell_material if spec and spec.shell_material else "-"],
        ["Подкладка", spec.lining_material if spec and spec.lining_material else "-"],
        ["Утеплитель", spec.insulation if spec and spec.insulation else "-"],
    ]
    spec_table = Table(spec_rows, colWidths=[55 * mm, 125 * mm])
    spec_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    flow.append(spec_table)

    if attribute_assignments:
        flow.append(Spacer(1, 4 * mm))
        flow.append(Paragraph("Атрибуты", heading_style))
        attr_rows = [["Атрибут", "Значение"]]
        for assignment in attribute_assignments:
            value_text = "-"
            if assignment.values:
                value_text = ", ".join(get_value_view(v, assignment.attribute.data_type) for v in assignment.values)
            attr_rows.append([assignment.attribute.name, value_text])
        attr_table = Table(attr_rows, colWidths=[65 * mm, 115 * mm], repeatRows=1)
        attr_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        flow.append(attr_table)

    flow.append(Spacer(1, 4 * mm))
    flow.append(Paragraph("Команда изделия", heading_style))
    team_rows = [
        ["Дизайнер", (product.designer.full_name or product.designer.login) if product.designer else "-"],
        ["Продукт менеджер", (product.product_manager.full_name or product.product_manager.login) if product.product_manager else "-"],
        ["Конструктор-модельер", (product.pattern_maker.full_name or product.pattern_maker.login) if product.pattern_maker else "-"],
        ["Технолог", (product.technologist.full_name or product.technologist.login) if product.technologist else "-"],
        ["Руководитель отдела", (product.department_head.full_name or product.department_head.login) if product.department_head else "-"],
    ]
    team_table = Table(team_rows, colWidths=[55 * mm, 125 * mm])
    team_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    flow.append(team_table)

    flow.append(Spacer(1, 4 * mm))
    flow.append(Paragraph("Задачи по изделию", heading_style))
    if tasks:
        task_rows = [["ID", "Название", "Статус", "Приоритет", "Очередь", "Дедлайн"]]
        for task in tasks:
            task_rows.append(
                [
                    str(task.id),
                    task.title,
                    TASK_STATUS_LABELS.get(task.status, task.status),
                    TASK_PRIORITY_LABELS.get(task.priority, task.priority),
                    task.queue.name if task.queue else "-",
                    task.deadline.isoformat() if task.deadline else "-",
                ]
            )
        task_table = Table(task_rows, colWidths=[10 * mm, 55 * mm, 30 * mm, 25 * mm, 35 * mm, 25 * mm], repeatRows=1)
        task_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        flow.append(task_table)
    else:
        flow.append(p("По этому изделию нет задач."))

    flow.append(Spacer(1, 4 * mm))
    flow.append(Paragraph("Файлы карточки", heading_style))
    if product.files:
        file_rows = [["Категория", "Название", "Файл"]]
        for f in product.files:
            file_rows.append(
                [
                    FILE_CATEGORY_LABELS.get(f.category, f.category),
                    f.title or "-",
                    f.original_name,
                ]
            )
        file_table = Table(file_rows, colWidths=[35 * mm, 65 * mm, 80 * mm], repeatRows=1)
        file_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        flow.append(file_table)
    else:
        flow.append(p("Файлы не добавлены."))

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"Product Card {product.sku}",
        author="FPLM",
    )
    doc.build(flow)
    return buf.getvalue()


@app.get("/qr/product/{product_id}.png")
def product_qr(product_id: int, db: Session = Depends(get_db)) -> Response:
    product = db.get(Product, product_id)
    if not product:
        return Response(content="", status_code=404)
    url = f"{_get_server_base_url()}/products/{product_id}"
    return Response(content=_qr_png(url), media_type="image/png")


@app.get("/qr/collection/{collection_id}.png")
def collection_qr(collection_id: int, db: Session = Depends(get_db)) -> Response:
    collection = db.get(Collection, collection_id)
    if not collection:
        return Response(content="", status_code=404)
    url = f"{_get_server_base_url()}/collections/{collection_id}"
    return Response(content=_qr_png(url), media_type="image/png")


def bootstrap_data() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_user_columns()
    _ensure_product_columns()
    _ensure_board_columns()
    _ensure_product_spec_columns()
    _ensure_uploads_dir()
    db = SessionLocal()
    try:
        legacy_role_map = {
            "content-manager": "user",
            "dictionary-manager": "admin",
            "read-only": "guest",
        }
        for existing_user in db.scalars(select(User)).all():
            mapped_role = legacy_role_map.get((existing_user.role or "").strip())
            if mapped_role and mapped_role != existing_user.role:
                existing_user.role = mapped_role

        if not db.scalar(select(User).where(User.login == "admin")):
            demo_users = [
                User(login="admin", password_hash=hash_password("admin"), role="admin", is_active=True),
                User(login="user", password_hash=hash_password("user"), role="user", is_active=True),
                User(login="guest", password_hash=hash_password("guest"), role="guest", is_active=True),
            ]
            db.add_all(demo_users)

        category_dict = db.scalar(select(Dictionary).where(Dictionary.code == "category"))
        if not category_dict:
            category_dict = Dictionary(code="category", name="Категории", description="Категории изделий")
            db.add(category_dict)
            db.flush()
            db.add_all(
                [
                    DictionaryItem(dictionary_id=category_dict.id, code="general", label="Общая", sort_order=1),
                    DictionaryItem(dictionary_id=category_dict.id, code="electronics", label="Электроника", sort_order=2),
                    DictionaryItem(dictionary_id=category_dict.id, code="mechanics", label="Механика", sort_order=3),
                ]
            )

        department_dict = db.scalar(select(Dictionary).where(Dictionary.code == "department"))
        if not department_dict:
            department_dict = Dictionary(code="department", name="Подразделения", description="Подразделения компании")
            db.add(department_dict)
            db.flush()
        existing_dept_codes = {
            c for c in db.scalars(select(DictionaryItem.code).where(DictionaryItem.dictionary_id == department_dict.id)).all()
        }
        for i, (code, label) in enumerate(
            [
                ("design", "Дизайн"),
                ("product", "Продуктовый отдел"),
                ("construction", "Конструкторский отдел"),
                ("technology", "Технологический отдел"),
                ("management", "Руководство"),
            ],
            start=1,
        ):
            if code not in existing_dept_codes:
                db.add(DictionaryItem(dictionary_id=department_dict.id, code=code, label=label, sort_order=i, is_active=True))

        position_dict = db.scalar(select(Dictionary).where(Dictionary.code == "position"))
        if not position_dict:
            position_dict = Dictionary(code="position", name="Должности", description="Справочник должностей")
            db.add(position_dict)
            db.flush()
        existing_position_codes = {
            c for c in db.scalars(select(DictionaryItem.code).where(DictionaryItem.dictionary_id == position_dict.id)).all()
        }
        for i, (code, label) in enumerate(
            [
                ("designer", "Дизайнер"),
                ("product_manager", "Продукт менеджер"),
                ("pattern_maker", "Конструктор-модельер"),
                ("technologist", "Технолог"),
                ("head", "Руководитель отдела"),
            ],
            start=1,
        ):
            if code not in existing_position_codes:
                db.add(DictionaryItem(dictionary_id=position_dict.id, code=code, label=label, sort_order=i, is_active=True))

        _normalize_spec_dictionaries(db)

        for field_name, cfg in SPEC_DICTIONARY_CONFIG.items():
            dict_code = str(cfg["code"])
            dict_name = str(cfg["name"])
            seed_items = [(str(x[0]), str(x[1])) for x in cfg.get("items", [])]
            dictionary = _resolve_spec_dictionary(db, field_name)
            if not dictionary:
                dictionary = Dictionary(code=dict_code, name=dict_name, description=f"Справочник для поля '{dict_name}'")
                db.add(dictionary)
                db.flush()
            existing_codes = {
                c for c in db.scalars(select(DictionaryItem.code).where(DictionaryItem.dictionary_id == dictionary.id)).all()
            }
            for sort_idx, (item_code, item_label) in enumerate(seed_items, start=1):
                if item_code not in existing_codes:
                    db.add(
                        DictionaryItem(
                            dictionary_id=dictionary.id,
                            code=item_code,
                            label=item_label,
                            sort_order=sort_idx,
                            is_active=True,
                        )
                    )

        if not db.scalar(select(Collection).limit(1)):
            db.add(Collection(code="FW26", name="Осень-Зима 2026", season="FW", year=2026, brand_line="Women Outerwear"))

        if not db.scalar(select(Supplier).limit(1)):
            db.add(Supplier(code="SUP-001", name="Nord Textile", country="Turkey", contact_email="sales@nord-textile.com"))

        if not db.scalar(select(TaskQueue).limit(1)):
            db.add(TaskQueue(code="FPLM", name="Fashion PLM", description="Основная очередь fashion-задач", is_active=True))

        if not db.scalar(select(TaskBoard).limit(1)):
            db.add(TaskBoard(code="DEV", name="Доска разработки", description="Канбан-доска проекта", is_active=True))

        existing_settings = {s.key for s in db.scalars(select(SystemSetting)).all()}
        for key, value in SETTINGS_DEFAULTS.items():
            if key not in existing_settings:
                db.add(SystemSetting(key=key, value=value))

        db.commit()
    finally:
        db.close()


bootstrap_data()


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    if request.session.get("user_id"):
        return _redirect("/products")
    return _redirect("/login")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.session.get("user_id"):
        return _redirect("/products")
    return _render(request, "login.html", {"title": "Р’С…РѕРґ"})


@app.post("/login")
def login(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.scalar(select(User).where(User.login == login))
    if not user or not user.is_active or not verify_password(password, user.password_hash):
        _set_flash(request, "Неверный логин или пароль", "error")
        return _redirect("/login")

    request.session["user_id"] = user.id
    _set_flash(request, "Вход выполнен", "success")
    return _redirect("/products")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return _redirect("/login")


@app.get("/cabinet", response_class=HTMLResponse)
def cabinet_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    authored_tasks = list(
        db.scalars(
            select(Task)
            .options(
                joinedload(Task.author),
                joinedload(Task.assignee),
                joinedload(Task.product),
                joinedload(Task.collection),
            )
            .where(Task.author_id == user.id)
            .order_by(Task.updated_at.desc())
        ).all()
    )
    assigned_tasks = list(
        db.scalars(
            select(Task)
            .options(
                joinedload(Task.author),
                joinedload(Task.assignee),
                joinedload(Task.product),
                joinedload(Task.collection),
            )
            .where(Task.assignee_id == user.id)
            .order_by(Task.updated_at.desc())
        ).all()
    )
    products = list(
        db.scalars(
            select(Product)
            .options(joinedload(Product.category_item))
            .where(
                or_(
                    Product.created_by == user.id,
                    Product.designer_id == user.id,
                    Product.product_manager_id == user.id,
                    Product.pattern_maker_id == user.id,
                    Product.technologist_id == user.id,
                    Product.department_head_id == user.id,
                )
            )
            .order_by(Product.updated_at.desc())
        ).all()
    )
    return _render(
        request,
        "cabinet.html",
        {
            "title": "Кабинет",
            "authored_tasks": authored_tasks,
            "assigned_tasks": assigned_tasks,
            "products": products,
            "user": user,
        },
    )


@app.get("/users", response_class=HTMLResponse)
def users_page(
    request: Request,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    users = list(
        db.scalars(
            select(User)
            .options(joinedload(User.department_item), joinedload(User.position_item))
            .order_by(User.created_at.desc())
        ).all()
    )
    _, department_items = _get_dictionary_items(db, "department")
    _, position_items = _get_dictionary_items(db, "position")
    return _render(
        request,
        "users/list.html",
        {
            "title": "Пользователи",
            "users": users,
            "roles": sorted(ROLES),
            "department_items": department_items,
            "position_items": position_items,
            "user": user,
        },
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    settings_map = {s.key: s.value for s in db.scalars(select(SystemSetting)).all()}
    values = {k: settings_map.get(k, v) for k, v in SETTINGS_DEFAULTS.items()}
    return _render(request, "settings.html", {"title": "Настройки системы", "values": values, "user": user})


@app.post("/settings")
def update_settings(
    request: Request,
    server_base_url: str = Form(...),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    incoming = {
        "server_base_url": server_base_url.strip() or SETTINGS_DEFAULTS["server_base_url"],
    }
    for key, val in incoming.items():
        row = db.scalar(select(SystemSetting).where(SystemSetting.key == key))
        if row:
            row.value = val
        else:
            db.add(SystemSetting(key=key, value=val))
    db.commit()
    _set_flash(request, "Настройки сохранены", "success")
    return _redirect("/settings")


@app.get("/users/{user_id}", response_class=HTMLResponse)
def user_detail_page(
    user_id: int,
    request: Request,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    target = db.scalar(
        select(User)
        .options(joinedload(User.department_item), joinedload(User.position_item))
        .where(User.id == user_id)
    )
    if not target:
        _set_flash(request, "Пользователь не найден", "error")
        return _redirect("/users")
    _, department_items = _get_dictionary_items(db, "department")
    _, position_items = _get_dictionary_items(db, "position")
    return _render(
        request,
        "users/detail.html",
        {
            "title": f"Пользователь: {target.login}",
            "target": target,
            "roles": sorted(ROLES),
            "department_items": department_items,
            "position_items": position_items,
            "user": user,
        },
    )


@app.post("/users")
def create_user(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(""),
    department_id_raw: str = Form(""),
    position_id_raw: str = Form(""),
    role: str = Form(...),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    login_clean = login.strip()
    password_clean = password.strip()
    if not login_clean:
        _set_flash(request, "Логин обязателен", "error")
        return _redirect("/users")
    if not password_clean:
        _set_flash(request, "Пароль обязателен", "error")
        return _redirect("/users")
    if role not in ROLES:
        _set_flash(request, "Некорректная роль", "error")
        return _redirect("/users")
    department_dict, _ = _get_dictionary_items(db, "department")
    position_dict, _ = _get_dictionary_items(db, "position")

    def parse_item_id(raw: str, dictionary: Dictionary | None) -> int | None:
        raw_clean = raw.strip()
        if not raw_clean:
            return None
        if not dictionary:
            raise ValueError("missing-dictionary")
        item_id: int | None = None
        if raw_clean.isdigit():
            item_id = int(raw_clean)
        else:
            item_id = db.scalar(
                select(DictionaryItem.id).where(
                    DictionaryItem.dictionary_id == dictionary.id,
                    DictionaryItem.is_active.is_(True),
                    or_(DictionaryItem.code == raw_clean, DictionaryItem.label == raw_clean),
                )
            )
        item_exists = (
            db.scalar(
                select(DictionaryItem.id).where(
                    DictionaryItem.id == item_id,
                    DictionaryItem.dictionary_id == dictionary.id,
                    DictionaryItem.is_active.is_(True),
                )
            )
            if item_id is not None
            else None
        )
        if item_exists is None:
            raise ValueError("bad-item")
        return int(item_exists)

    try:
        department_item_id = parse_item_id(department_id_raw, department_dict)
        position_item_id = parse_item_id(position_id_raw, position_dict)
    except ValueError:
        _set_flash(request, "Некорректное подразделение или должность", "error")
        return _redirect("/users")

    department_label = db.scalar(select(DictionaryItem.label).where(DictionaryItem.id == department_item_id)) if department_item_id else None
    position_label = db.scalar(select(DictionaryItem.label).where(DictionaryItem.id == position_item_id)) if position_item_id else None

    db.add(
        User(
            login=login_clean,
            password_hash=hash_password(password_clean),
            full_name=full_name.strip() or None,
            department=department_label,
            position=position_label,
            department_item_id=department_item_id,
            position_item_id=position_item_id,
            role=role,
            is_active=active is not None,
        )
    )
    try:
        db.commit()
        _set_flash(request, "Пользователь создан", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "Логин уже используется", "error")
    return _redirect("/users")


@app.post("/users/{user_id}/toggle")
def toggle_user(
    user_id: int,
    request: Request,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if target:
        if target.id == user.id:
            _set_flash(request, "Нельзя блокировать собственную учетную запись", "error")
            return _redirect("/users")
        if target.role == "admin" and target.is_active:
            remaining = db.scalar(
                select(func.count(User.id)).where(
                    User.role == "admin",
                    User.is_active.is_(True),
                    User.id != target.id,
                )
            )
            if not remaining:
                _set_flash(request, "Нельзя блокировать последнего активного администратора", "error")
                return _redirect("/users")
        target.is_active = not target.is_active
        db.commit()
        _set_flash(request, "Статус пользователя обновлен", "success")
    return _redirect("/users")


@app.post("/users/{user_id}/role")
def change_role(
    user_id: int,
    request: Request,
    role: str = Form(...),
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if not target:
        _set_flash(request, "Пользователь не найден", "error")
        return _redirect("/users")
    if role not in ROLES:
        _set_flash(request, "Некорректная роль", "error")
        return _redirect("/users")
    if target and role in ROLES:
        if target.id == user.id:
            _set_flash(request, "Нельзя менять роль собственной учетной записи", "error")
            return _redirect("/users")
        if target.role == "admin" and role != "admin":
            remaining = db.scalar(
                select(func.count(User.id)).where(
                    User.role == "admin",
                    User.is_active.is_(True),
                    User.id != target.id,
                )
            )
            if not remaining:
                _set_flash(request, "Нельзя снять роль у последнего активного администратора", "error")
                return _redirect("/users")
        target.role = role
        db.commit()
        _set_flash(request, "Роль пользователя обновлена", "success")
    return _redirect("/users")


@app.post("/users/{user_id}/profile")
def update_user_profile(
    user_id: int,
    request: Request,
    full_name: str = Form(""),
    department_id_raw: str = Form(""),
    department_item_id_raw: str = Form(""),
    department: str = Form(""),
    position_id_raw: str = Form(""),
    position_item_id_raw: str = Form(""),
    position: str = Form(""),
    return_to: str = Form(""),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    redirect_to = (return_to or request.headers.get("referer") or f"/users/{user_id}").strip()
    if not redirect_to.startswith("/"):
        redirect_to = f"/users/{user_id}"

    target = db.get(User, user_id)
    if not target:
        _set_flash(request, "Пользователь не найден", "error")
        return _redirect("/users")
    department_dict, _ = _get_dictionary_items(db, "department")
    position_dict, _ = _get_dictionary_items(db, "position")

    def resolve_item(raw: str, dictionary: Dictionary | None) -> tuple[int | None, str | None]:
        raw_clean = raw.strip()
        if not raw_clean or not dictionary:
            return None, None
        item_id: int | None = None
        if raw_clean.isdigit():
            item_id = int(raw_clean)
        else:
            item_id = db.scalar(
                select(DictionaryItem.id).where(
                    DictionaryItem.dictionary_id == dictionary.id,
                    DictionaryItem.is_active.is_(True),
                    or_(DictionaryItem.code == raw_clean, DictionaryItem.label == raw_clean),
                )
            )
        if item_id is None:
            return None, None
        item = db.scalar(
            select(DictionaryItem).where(
                DictionaryItem.id == item_id,
                DictionaryItem.dictionary_id == dictionary.id,
                DictionaryItem.is_active.is_(True),
            )
        )
        if not item:
            return None, None
        return item.id, item.label

    department_raw = department_id_raw if department_id_raw.strip() else department_item_id_raw
    position_raw = position_id_raw if position_id_raw.strip() else position_item_id_raw
    department_item_id, department_label = resolve_item(department_raw, department_dict)
    position_item_id, position_label = resolve_item(position_raw, position_dict)
    if not department_label and department.strip():
        department_label = department.strip()
    if not position_label and position.strip():
        position_label = position.strip()
    target.full_name = full_name.strip() or None
    target.department = department_label
    target.position = position_label
    target.department_item_id = department_item_id
    target.position_item_id = position_item_id
    try:
        db.commit()
    except Exception:
        db.rollback()
        _set_flash(request, "Не удалось сохранить изменения пользователя", "error")
        return _redirect(redirect_to)
    _set_flash(request, "Профиль пользователя обновлен", "success")
    return _redirect(redirect_to)


@app.post("/users/{user_id}/delete")
def delete_user(
    user_id: int,
    request: Request,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if not target:
        _set_flash(request, "Пользователь не найден", "error")
        return _redirect("/users")
    if target.id == user.id:
        _set_flash(request, "Нельзя удалить собственную учетную запись", "error")
        return _redirect("/users")
    if target.role == "admin" and target.is_active:
        remaining = db.scalar(
            select(func.count(User.id)).where(
                User.role == "admin",
                User.is_active.is_(True),
                User.id != target.id,
            )
        )
        if not remaining:
            _set_flash(request, "Нельзя удалить последнего активного администратора", "error")
            return _redirect("/users")

    usage = _user_usage_counts(db, target.id)
    if any(v > 0 for v in usage.values()):
        reasons: list[str] = []
        if usage["products_team"]:
            reasons.append(f"изделия (команда, {usage['products_team']})")
        if usage["tasks_owner"]:
            reasons.append(f"задачи ({usage['tasks_owner']})")
        _set_flash(request, f"Нельзя удалить пользователя: есть связанные данные ({', '.join(reasons)}).", "error")
        return _redirect("/users")

    db.query(Product).filter(Product.created_by == target.id).update({Product.created_by: None}, synchronize_session=False)
    db.query(Product).filter(Product.updated_by == target.id).update({Product.updated_by: None}, synchronize_session=False)
    db.query(ProductFile).filter(ProductFile.uploaded_by == target.id).update(
        {ProductFile.uploaded_by: None}, synchronize_session=False
    )
    db.query(TaskFile).filter(TaskFile.uploaded_by == target.id).update({TaskFile.uploaded_by: None}, synchronize_session=False)

    db.delete(target)
    db.commit()
    _set_flash(request, "Пользователь удален", "success")
    return _redirect("/users")


@app.get("/queues", response_class=HTMLResponse)
def queues_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    queues = list(db.scalars(select(TaskQueue).order_by(TaskQueue.name.asc())).all())
    return _render(
        request,
        "tasks/queues.html",
        {
            "title": "Очереди задач",
            "queues": queues,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.get("/queues/{queue_id}", response_class=HTMLResponse)
def queue_detail_page(
    queue_id: int,
    request: Request,
    q: str = "",
    status: str = "",
    priority: str = "",
    assignee_id_raw: str = "",
    collection_id_raw: str = "",
    product_id_raw: str = "",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    queue = db.get(TaskQueue, queue_id)
    if not queue:
        _set_flash(request, "Очередь не найдена", "error")
        return _redirect("/queues")
    related_board = db.scalar(
        select(TaskBoard)
        .where(TaskBoard.filter_queue_id == queue.id, TaskBoard.is_active.is_(True))
        .order_by(TaskBoard.name.asc())
        .limit(1)
    )

    assignee_id = int(assignee_id_raw) if assignee_id_raw.strip().isdigit() else None
    collection_id = int(collection_id_raw) if collection_id_raw.strip().isdigit() else None
    product_id = int(product_id_raw) if product_id_raw.strip().isdigit() else None

    stmt = (
        select(Task)
        .options(
            joinedload(Task.author),
            joinedload(Task.assignee),
            joinedload(Task.collection),
            joinedload(Task.product),
        )
        .where(Task.queue_id == queue_id)
    )
    stmt = _apply_task_filters(stmt, q, status, priority, None, assignee_id, collection_id, product_id).order_by(
        Task.created_at.desc()
    )
    tasks = list(db.scalars(stmt).all())
    users = list(db.scalars(select(User).where(User.is_active.is_(True)).order_by(User.login.asc())).all())
    products = list(db.scalars(select(Product).order_by(Product.name.asc())).all())
    collections = list(db.scalars(select(Collection).where(Collection.is_active.is_(True)).order_by(Collection.year.desc())).all())

    return _render(
        request,
        "tasks/queue_detail.html",
        {
            "title": f"Очередь: {queue.name}",
            "queue": queue,
            "related_board": related_board,
            "tasks": tasks,
            "users": users,
            "products": products,
            "collections": collections,
            "product_collection_map": _product_collection_map(db),
            "status_order": TASK_STATUS_ORDER,
            "priorities": TASK_PRIORITIES,
            "q": q,
            "selected_status": status,
            "selected_priority": priority,
            "selected_assignee_id": assignee_id,
            "selected_collection_id": collection_id,
            "selected_product_id": product_id,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/queues")
def create_queue(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    name_clean = name.strip()
    if not name_clean:
        _set_flash(request, "Название очереди обязательно", "error")
        return _redirect("/queues")
    created = False
    for _ in range(5):
        existing_codes = list(db.scalars(select(TaskQueue.code)).all())
        next_code = _next_auto_code(existing_codes, "Q")
        db.add(
            TaskQueue(
                code=next_code,
                name=name_clean,
                description=description.strip() or None,
                is_active=active is not None,
            )
        )
        try:
            db.commit()
            _set_flash(request, f"Очередь создана ({next_code})", "success")
            created = True
            break
        except IntegrityError:
            db.rollback()
    if not created:
        _set_flash(request, "Не удалось создать очередь: повторите попытку", "error")
    return _redirect("/queues")


@app.post("/queues/{queue_id}/update")
def update_queue(
    queue_id: int,
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    name_clean = name.strip()
    if not name_clean:
        _set_flash(request, "Название очереди обязательно", "error")
        return _redirect("/queues")
    queue = db.get(TaskQueue, queue_id)
    if queue:
        new_is_active = active is not None
        if queue.is_active and not new_is_active and _queue_usage_count(db, queue.id):
            _set_flash(request, "Нельзя деактивировать очередь: она используется в задачах", "error")
            return _redirect("/queues")

        queue.name = name_clean
        queue.description = description.strip() or None
        queue.is_active = new_is_active
        try:
            db.commit()
            _set_flash(request, "Очередь обновлена", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код очереди должен быть уникален", "error")
    return _redirect("/queues")


@app.post("/queues/{queue_id}/delete")
def delete_queue(
    queue_id: int,
    request: Request,
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    queue = db.get(TaskQueue, queue_id)
    if not queue:
        return _redirect("/queues")

    task_usage = _queue_usage_count(db, queue.id)
    board_usage = _queue_board_usage_count(db, queue.id)
    if task_usage or board_usage:
        reasons: list[str] = []
        if task_usage:
            reasons.append(f"задачи ({task_usage})")
        if board_usage:
            reasons.append(f"доски ({board_usage})")
        _set_flash(request, f"Нельзя удалить очередь: есть связанные данные ({', '.join(reasons)}).", "error")
        return _redirect(f"/queues/{queue_id}")

    db.delete(queue)
    db.commit()
    _set_flash(request, "Очередь удалена", "success")
    return _redirect("/queues")


@app.get("/boards", response_class=HTMLResponse)
def boards_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    boards = list(db.scalars(select(TaskBoard).options(joinedload(TaskBoard.filter_queue)).order_by(TaskBoard.name.asc())).all())
    queues = list(db.scalars(select(TaskQueue).where(TaskQueue.is_active.is_(True)).order_by(TaskQueue.name.asc())).all())
    return _render(
        request,
        "tasks/boards.html",
        {
            "title": "Доски задач",
            "boards": boards,
            "queues": queues,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.get("/boards/{board_id}", response_class=HTMLResponse)
def board_detail_page(
    board_id: int,
    request: Request,
    q: str = "",
    status: str = "",
    priority: str = "",
    queue_id_raw: str = "",
    assignee_id_raw: str = "",
    collection_id_raw: str = "",
    product_id_raw: str = "",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    board = db.scalar(select(TaskBoard).options(joinedload(TaskBoard.filter_queue)).where(TaskBoard.id == board_id))
    if not board:
        _set_flash(request, "Доска не найдена", "error")
        return _redirect("/boards")
    queue_id = int(queue_id_raw) if queue_id_raw.strip().isdigit() else board.filter_queue_id
    assignee_id = int(assignee_id_raw) if assignee_id_raw.strip().isdigit() else None
    collection_id = int(collection_id_raw) if collection_id_raw.strip().isdigit() else None
    product_id = int(product_id_raw) if product_id_raw.strip().isdigit() else None

    stmt = select(Task).options(
        joinedload(Task.author),
        joinedload(Task.assignee),
        joinedload(Task.queue),
        joinedload(Task.collection),
        joinedload(Task.product),
    )
    stmt = _apply_task_filters(stmt, q, status, priority, queue_id, assignee_id, collection_id, product_id).order_by(
        Task.created_at.desc()
    )
    tasks = list(db.scalars(stmt).all())
    queues = list(db.scalars(select(TaskQueue).where(TaskQueue.is_active.is_(True)).order_by(TaskQueue.name.asc())).all())
    users = list(db.scalars(select(User).where(User.is_active.is_(True)).order_by(User.login.asc())).all())
    products = list(db.scalars(select(Product).order_by(Product.name.asc())).all())
    collections = list(db.scalars(select(Collection).where(Collection.is_active.is_(True)).order_by(Collection.year.desc())).all())
    return _render(
        request,
        "tasks/board_detail.html",
        {
            "title": f"Доска: {board.name}",
            "board": board,
            "tasks": tasks,
            "queues": queues,
            "users": users,
            "products": products,
            "collections": collections,
            "product_collection_map": _product_collection_map(db),
            "status_order": TASK_STATUS_ORDER,
            "priorities": TASK_PRIORITIES,
            "q": q,
            "selected_status": status,
            "selected_priority": priority,
            "selected_queue_id": queue_id,
            "selected_assignee_id": assignee_id,
            "selected_collection_id": collection_id,
            "selected_product_id": product_id,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/boards")
def create_board(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    filter_queue_id_raw: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    name_clean = name.strip()
    if not name_clean:
        _set_flash(request, "Название доски обязательно", "error")
        return _redirect("/boards")
    created = False
    filter_queue_id = int(filter_queue_id_raw) if filter_queue_id_raw.strip().isdigit() else None
    if filter_queue_id is not None and not db.scalar(
        select(TaskQueue.id).where(TaskQueue.id == filter_queue_id, TaskQueue.is_active.is_(True))
    ):
        _set_flash(request, "Некорректная очередь для фильтра доски", "error")
        return _redirect("/boards")
    for _ in range(5):
        existing_codes = list(db.scalars(select(TaskBoard.code)).all())
        next_code = _next_auto_code(existing_codes, "K")
        db.add(
            TaskBoard(
                code=next_code,
                name=name_clean,
                description=description.strip() or None,
                filter_queue_id=filter_queue_id,
                is_active=active is not None,
            )
        )
        try:
            db.commit()
            _set_flash(request, f"Доска создана ({next_code})", "success")
            created = True
            break
        except IntegrityError:
            db.rollback()
    if not created:
        _set_flash(request, "Не удалось создать доску: повторите попытку", "error")
    return _redirect("/boards")


@app.post("/boards/{board_id}/update")
def update_board(
    board_id: int,
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    filter_queue_id_raw: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    name_clean = name.strip()
    if not name_clean:
        _set_flash(request, "Название доски обязательно", "error")
        return _redirect(f"/boards/{board_id}")
    board = db.scalar(select(TaskBoard).options(joinedload(TaskBoard.filter_queue)).where(TaskBoard.id == board_id))
    if board:
        filter_queue_id = int(filter_queue_id_raw) if filter_queue_id_raw.strip().isdigit() else None
        if filter_queue_id is not None and not db.scalar(
            select(TaskQueue.id).where(TaskQueue.id == filter_queue_id, TaskQueue.is_active.is_(True))
        ):
            _set_flash(request, "Некорректная очередь для фильтра доски", "error")
            return _redirect(f"/boards/{board_id}")
        board.name = name_clean
        board.description = description.strip() or None
        board.filter_queue_id = filter_queue_id
        board.is_active = active is not None
        try:
            db.commit()
            _set_flash(request, "Доска обновлена", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код доски должен быть уникален", "error")
    return _redirect("/boards")


@app.post("/boards/{board_id}/delete")
def delete_board(
    board_id: int,
    request: Request,
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    board = db.get(TaskBoard, board_id)
    if not board:
        return _redirect("/boards")

    usage = _board_usage_count(db, board.id)
    if usage:
        _set_flash(request, f"Нельзя удалить доску: есть связанные задачи ({usage}).", "error")
        return _redirect(f"/boards/{board_id}")

    db.delete(board)
    db.commit()
    _set_flash(request, "Доска удалена", "success")
    return _redirect("/boards")


@app.get("/boards/{board_id}/kanban", response_class=HTMLResponse)
def board_kanban(
    board_id: int,
    request: Request,
    q: str = "",
    status_filter: str = "",
    priority: str = "",
    queue_id_raw: str = "",
    assignee_id_raw: str = "",
    collection_id_raw: str = "",
    product_id_raw: str = "",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    board = db.scalar(select(TaskBoard).options(joinedload(TaskBoard.filter_queue)).where(TaskBoard.id == board_id))
    if not board:
        _set_flash(request, "Доска не найдена", "error")
        return _redirect("/boards")

    queue_id = int(queue_id_raw) if queue_id_raw.strip().isdigit() else board.filter_queue_id
    assignee_id = int(assignee_id_raw) if assignee_id_raw.strip().isdigit() else None
    collection_id = int(collection_id_raw) if collection_id_raw.strip().isdigit() else None
    product_id = int(product_id_raw) if product_id_raw.strip().isdigit() else None

    stmt = (
        select(Task)
        .options(
            joinedload(Task.author),
            joinedload(Task.assignee),
            joinedload(Task.product),
            joinedload(Task.collection),
            joinedload(Task.queue),
        )
    )
    stmt = _apply_task_filters(stmt, q, status_filter, priority, queue_id, assignee_id, collection_id, product_id).order_by(
        Task.priority.desc(), Task.created_at.desc()
    )
    tasks = list(db.scalars(stmt).all())
    grouped: dict[str, list[Task]] = {k: [] for k in TASK_STATUS_ORDER}
    for task in tasks:
        grouped.setdefault(task.status, []).append(task)
    queues = list(db.scalars(select(TaskQueue).where(TaskQueue.is_active.is_(True)).order_by(TaskQueue.name.asc())).all())
    users = list(db.scalars(select(User).where(User.is_active.is_(True)).order_by(User.login.asc())).all())
    products = list(db.scalars(select(Product).order_by(Product.name.asc())).all())
    collections = list(db.scalars(select(Collection).where(Collection.is_active.is_(True)).order_by(Collection.year.desc())).all())

    return _render(
        request,
        "tasks/kanban.html",
        {
            "title": f"Канбан: {board.name}",
            "board": board,
            "grouped": grouped,
            "status_order": TASK_STATUS_ORDER,
            "priorities": TASK_PRIORITIES,
            "queues": queues,
            "users": users,
            "products": products,
            "collections": collections,
            "product_collection_map": _product_collection_map(db),
            "q": q,
            "selected_status": status_filter,
            "selected_priority": priority,
            "selected_queue_id": queue_id,
            "selected_assignee_id": assignee_id,
            "selected_collection_id": collection_id,
            "selected_product_id": product_id,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.get("/tasks", response_class=HTMLResponse)
def tasks_page(
    _: Request,
    __: User = Depends(get_current_user),
):
    return _redirect("/queues")


@app.post("/tasks")
def create_task(
    request: Request,
    title: str = Form(...),
    comment: str = Form(""),
    status: str = Form("todo"),
    priority: str = Form("medium"),
    tags: str = Form(""),
    start_date: str = Form(""),
    end_date: str = Form(""),
    deadline: str = Form(""),
    assignee_id_raw: str = Form(""),
    queue_id_raw: str = Form(""),
    collection_id_raw: str = Form(""),
    product_id_raw: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    title_clean = title.strip()
    if not title_clean:
        _set_flash(request, "Название задачи обязательно", "error")
        return _redirect(request.headers.get("referer") or "/queues")
    if status not in TASK_STATUS_ORDER:
        _set_flash(request, "Некорректный статус задачи", "error")
        return _redirect(request.headers.get("referer") or "/queues")
    if priority not in TASK_PRIORITIES:
        _set_flash(request, "Некорректный приоритет задачи", "error")
        return _redirect(request.headers.get("referer") or "/queues")

    def parse_date(raw: str):
        return datetime.fromisoformat(raw).date() if raw.strip() else None

    def parse_int(raw: str):
        return int(raw) if raw.strip().isdigit() else None

    try:
        parsed_start_date = parse_date(start_date)
        parsed_end_date = parse_date(end_date)
        parsed_deadline = parse_date(deadline)
    except ValueError:
        _set_flash(request, "Проверьте даты задачи (формат YYYY-MM-DD)", "error")
        return _redirect(request.headers.get("referer") or "/queues")

    parsed_assignee_id = parse_int(assignee_id_raw)
    parsed_queue_id = parse_int(queue_id_raw)
    parsed_product_id = parse_int(product_id_raw)
    parsed_collection_id = parse_int(collection_id_raw)
    resolved_collection_id = _resolve_task_collection_id(db, parsed_product_id, parsed_collection_id)

    task = Task(
        title=title_clean,
        comment=comment.strip() or None,
        status=status,
        priority=priority,
        tags=tags.strip() or None,
        start_date=parsed_start_date,
        end_date=parsed_end_date,
        deadline=parsed_deadline,
        author_id=user.id,
        assignee_id=parsed_assignee_id,
        queue_id=parsed_queue_id,
        board_id=None,
        collection_id=resolved_collection_id,
        product_id=parsed_product_id,
    )

    if task.queue_id is None:
        _set_flash(request, "Задача должна принадлежать очереди", "error")
        return _redirect(request.headers.get("referer") or "/queues")
    if not db.scalar(select(TaskQueue.id).where(TaskQueue.id == task.queue_id, TaskQueue.is_active.is_(True))):
        _set_flash(request, "Выбрана несуществующая или неактивная очередь", "error")
        return _redirect(request.headers.get("referer") or "/queues")
    if task.assignee_id is not None and not db.scalar(
        select(User.id).where(User.id == task.assignee_id, User.is_active.is_(True))
    ):
        _set_flash(request, "Выбран несуществующий или неактивный исполнитель", "error")
        return _redirect(request.headers.get("referer") or "/queues")
    if task.collection_id is not None and not db.scalar(
        select(Collection.id).where(Collection.id == task.collection_id, Collection.is_active.is_(True))
    ):
        _set_flash(request, "Выбрана несуществующая или неактивная коллекция", "error")
        return _redirect(request.headers.get("referer") or "/queues")
    if task.product_id is not None and not db.scalar(select(Product.id).where(Product.id == task.product_id)):
        _set_flash(request, "Выбрано несуществующее изделие", "error")
        return _redirect(request.headers.get("referer") or "/queues")

    db.add(task)
    db.commit()
    _set_flash(request, "Задача создана", "success")
    return _redirect(f"/tasks/{task.id}")


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
def task_detail(
    task_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.scalar(
        select(Task)
        .options(
            joinedload(Task.author),
            joinedload(Task.assignee),
            joinedload(Task.queue),
            joinedload(Task.product),
            joinedload(Task.collection),
            selectinload(Task.files).joinedload(TaskFile.uploader),
        )
        .where(Task.id == task_id)
    )
    if not task:
        _set_flash(request, "Задача не найдена", "error")
        return _redirect("/tasks")

    queues = list(db.scalars(select(TaskQueue).where(TaskQueue.is_active.is_(True)).order_by(TaskQueue.name.asc())).all())
    users = list(db.scalars(select(User).where(User.is_active.is_(True)).order_by(User.login.asc())).all())
    products = list(db.scalars(select(Product).order_by(Product.name.asc())).all())
    collections = list(db.scalars(select(Collection).where(Collection.is_active.is_(True)).order_by(Collection.year.desc())).all())

    return _render(
        request,
        "tasks/detail.html",
        {
            "title": f"Задача: {task.title}",
            "task": task,
            "queues": queues,
            "users": users,
            "products": products,
            "collections": collections,
            "product_collection_map": _product_collection_map(db),
            "status_order": TASK_STATUS_ORDER,
            "priorities": TASK_PRIORITIES,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/tasks/{task_id}/update")
def update_task(
    task_id: int,
    request: Request,
    title: str = Form(...),
    comment: str = Form(""),
    status: str = Form("todo"),
    priority: str = Form("medium"),
    tags: str = Form(""),
    start_date: str = Form(""),
    end_date: str = Form(""),
    deadline: str = Form(""),
    assignee_id_raw: str = Form(""),
    queue_id_raw: str = Form(""),
    collection_id_raw: str = Form(""),
    product_id_raw: str = Form(""),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if not task:
        return _redirect("/tasks")
    title_clean = title.strip()
    if not title_clean:
        _set_flash(request, "Название задачи обязательно", "error")
        return _redirect(f"/tasks/{task_id}")
    if status not in TASK_STATUS_ORDER or priority not in TASK_PRIORITIES:
        _set_flash(request, "Некорректный статус или приоритет", "error")
        return _redirect(f"/tasks/{task_id}")

    def parse_date(raw: str):
        return datetime.fromisoformat(raw).date() if raw.strip() else None

    def parse_int(raw: str):
        return int(raw) if raw.strip().isdigit() else None

    try:
        task.title = title_clean
        task.comment = comment.strip() or None
        task.status = status
        task.priority = priority
        task.tags = tags.strip() or None
        task.start_date = parse_date(start_date)
        task.end_date = parse_date(end_date)
        task.deadline = parse_date(deadline)
        parsed_product_id = parse_int(product_id_raw)
        parsed_collection_id = parse_int(collection_id_raw)
        task.assignee_id = parse_int(assignee_id_raw)
        task.queue_id = parse_int(queue_id_raw)
        task.board_id = None
        task.product_id = parsed_product_id
        task.collection_id = _resolve_task_collection_id(db, parsed_product_id, parsed_collection_id)
    except ValueError:
        _set_flash(request, "Проверьте даты задачи (формат YYYY-MM-DD)", "error")
        return _redirect(f"/tasks/{task_id}")

    if task.queue_id is None:
        _set_flash(request, "Задача должна принадлежать очереди", "error")
        return _redirect(f"/tasks/{task_id}")
    if not db.scalar(select(TaskQueue.id).where(TaskQueue.id == task.queue_id, TaskQueue.is_active.is_(True))):
        _set_flash(request, "Выбрана несуществующая или неактивная очередь", "error")
        return _redirect(f"/tasks/{task_id}")
    if task.assignee_id is not None and not db.scalar(
        select(User.id).where(User.id == task.assignee_id, User.is_active.is_(True))
    ):
        _set_flash(request, "Выбран несуществующий или неактивный исполнитель", "error")
        return _redirect(f"/tasks/{task_id}")
    if task.collection_id is not None and not db.scalar(
        select(Collection.id).where(Collection.id == task.collection_id, Collection.is_active.is_(True))
    ):
        _set_flash(request, "Выбрана несуществующая или неактивная коллекция", "error")
        return _redirect(f"/tasks/{task_id}")
    if task.product_id is not None and not db.scalar(select(Product.id).where(Product.id == task.product_id)):
        _set_flash(request, "Выбрано несуществующее изделие", "error")
        return _redirect(f"/tasks/{task_id}")

    db.commit()
    _set_flash(request, "Задача обновлена", "success")
    return _redirect(f"/tasks/{task_id}")


@app.post("/tasks/{task_id}/status")
def update_task_status(
    task_id: int,
    request: Request,
    status: str = Form(...),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if task and status in TASK_STATUS_ORDER:
        task.status = status
        db.commit()
        _set_flash(request, "Статус задачи обновлен", "success")
    return _redirect(request.headers.get("referer") or "/tasks")


@app.post("/tasks/{task_id}/status-json")
def update_task_status_json(
    task_id: int,
    status: str = Form(...),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if not task:
        return JSONResponse({"ok": False, "error": "task_not_found"}, status_code=404)
    if status not in TASK_STATUS_ORDER:
        return JSONResponse({"ok": False, "error": "invalid_status"}, status_code=400)
    task.status = status
    db.commit()
    return JSONResponse({"ok": True, "task_id": task_id, "status": status})


@app.post("/tasks/{task_id}/files")
def upload_task_file(
    task_id: int,
    request: Request,
    task_file: UploadFile = File(...),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if not task:
        return _redirect("/tasks")

    suffix = Path(task_file.filename or "file.bin").suffix.lower()
    allowed = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".csv",
        ".txt",
        ".zip",
        ".rar",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".svg",
        ".dxf",
        ".plt",
    }
    if suffix and suffix not in allowed:
        _set_flash(request, "Недопустимый тип файла", "error")
        return _redirect(f"/tasks/{task_id}")

    _ensure_uploads_dir()
    file_name = f"tf_{task_id}_{uuid4().hex[:12]}{suffix or '.bin'}"
    dst = TASK_FILES_DIR / file_name
    data = task_file.file.read()
    if not data:
        _set_flash(request, "Пустой файл", "error")
        return _redirect(f"/tasks/{task_id}")
    dst.write_bytes(data)

    db.add(
        TaskFile(
            task_id=task_id,
            original_name=task_file.filename or file_name,
            file_path=f"/static/task_files/{file_name}",
            mime_type=task_file.content_type,
            uploaded_by=user.id,
        )
    )
    db.commit()
    _set_flash(request, "Файл прикреплен к задаче", "success")
    return _redirect(f"/tasks/{task_id}")


@app.post("/tasks/{task_id}/files/{file_id}/delete")
def delete_task_file(
    task_id: int,
    file_id: int,
    request: Request,
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    rec = db.scalar(select(TaskFile).where(TaskFile.id == file_id, TaskFile.task_id == task_id))
    if not rec:
        return _redirect(f"/tasks/{task_id}")
    if rec.file_path.startswith("/static/task_files/"):
        file_name = rec.file_path.split("/static/task_files/", 1)[1]
        path = TASK_FILES_DIR / file_name
        if path.exists():
            path.unlink()
    db.delete(rec)
    db.commit()
    _set_flash(request, "Файл задачи удален", "success")
    return _redirect(f"/tasks/{task_id}")


@app.get("/collections", response_class=HTMLResponse)
def collections_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    collections = list(db.scalars(select(Collection).order_by(Collection.year.desc(), Collection.code.asc())).all())
    return _render(
        request,
        "collections/list.html",
        {
            "title": "Коллекции",
            "collections": collections,
            "product_collection_map": _product_collection_map(db),
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.get("/collections/{collection_id}", response_class=HTMLResponse)
def collection_detail_page(
    collection_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    collection = db.get(Collection, collection_id)
    if not collection:
        _set_flash(request, "Коллекция не найдена", "error")
        return _redirect("/collections")
    products = list(
        db.scalars(
            select(Product)
            .join(ProductSpec, ProductSpec.product_id == Product.id)
            .where(ProductSpec.collection_id == collection_id)
            .order_by(Product.updated_at.desc())
        ).all()
    )
    return _render(
        request,
        "collections/detail.html",
        {
            "title": f"Коллекция: {collection.name}",
            "collection": collection,
            "products": products,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/collections")
def create_collection(
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    season: str = Form(...),
    year: int = Form(...),
    brand_line: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    code_clean = code.strip()
    name_clean = name.strip()
    season_clean = season.strip().upper()
    if not code_clean or not name_clean:
        _set_flash(request, "Код и название коллекции обязательны", "error")
        return _redirect("/collections")
    if season_clean not in {"FW", "SS"}:
        _set_flash(request, "Сезон должен быть FW или SS", "error")
        return _redirect("/collections")
    if year < 2000 or year > 2100:
        _set_flash(request, "Год коллекции должен быть в диапазоне 2000-2100", "error")
        return _redirect("/collections")
    db.add(
        Collection(
            code=code_clean,
            name=name_clean,
            season=season_clean,
            year=year,
            brand_line=brand_line.strip() or None,
            is_active=active is not None,
        )
    )
    try:
        db.commit()
        _set_flash(request, "Коллекция создана", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "Код коллекции должен быть уникален", "error")
    return _redirect("/collections")


@app.post("/collections/{collection_id}/update")
def update_collection(
    collection_id: int,
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    season: str = Form(...),
    year: int = Form(...),
    brand_line: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    code_clean = code.strip()
    name_clean = name.strip()
    season_clean = season.strip().upper()
    if not code_clean or not name_clean:
        _set_flash(request, "Код и название коллекции обязательны", "error")
        return _redirect("/collections")
    if season_clean not in {"FW", "SS"}:
        _set_flash(request, "Сезон должен быть FW или SS", "error")
        return _redirect("/collections")
    if year < 2000 or year > 2100:
        _set_flash(request, "Год коллекции должен быть в диапазоне 2000-2100", "error")
        return _redirect("/collections")
    collection = db.get(Collection, collection_id)
    if collection:
        new_is_active = active is not None
        if collection.is_active and not new_is_active:
            usage = _collection_usage_count(db, collection.id)
            if usage:
                _set_flash(request, "Нельзя деактивировать коллекцию: она используется в изделиях", "error")
                return _redirect("/collections")

        collection.code = code_clean
        collection.name = name_clean
        collection.season = season_clean
        collection.year = year
        collection.brand_line = brand_line.strip() or None
        collection.is_active = new_is_active
        try:
            db.commit()
            _set_flash(request, "Коллекция обновлена", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код коллекции должен быть уникален", "error")
    return _redirect("/collections")


@app.post("/collections/{collection_id}/delete")
def delete_collection(
    collection_id: int,
    request: Request,
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    collection = db.get(Collection, collection_id)
    if not collection:
        return _redirect("/collections")

    usage = _collection_usage_count(db, collection.id)
    if usage:
        _set_flash(request, f"Нельзя удалить коллекцию: есть связанные изделия ({usage}).", "error")
        return _redirect(f"/collections/{collection_id}")

    db.delete(collection)
    db.commit()
    _set_flash(request, "Коллекция удалена", "success")
    return _redirect("/collections")


@app.get("/suppliers", response_class=HTMLResponse)
def suppliers_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suppliers = list(db.scalars(select(Supplier).order_by(Supplier.name.asc())).all())
    return _render(
        request,
        "suppliers/list.html",
        {
            "title": "Поставщики",
            "suppliers": suppliers,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.get("/suppliers/{supplier_id}", response_class=HTMLResponse)
def supplier_detail_page(
    supplier_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    supplier = db.get(Supplier, supplier_id)
    if not supplier:
        _set_flash(request, "Поставщик не найден", "error")
        return _redirect("/suppliers")
    products = list(
        db.scalars(
            select(Product)
            .join(ProductSpec, ProductSpec.product_id == Product.id)
            .where(ProductSpec.supplier_id == supplier_id)
            .order_by(Product.updated_at.desc())
        ).all()
    )
    return _render(
        request,
        "suppliers/detail.html",
        {
            "title": f"Поставщик: {supplier.name}",
            "supplier": supplier,
            "products": products,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/suppliers")
def create_supplier(
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    country: str = Form(...),
    contact_email: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    code_clean = code.strip()
    name_clean = name.strip()
    country_clean = country.strip()
    email_clean = contact_email.strip()
    if not code_clean or not name_clean or not country_clean:
        _set_flash(request, "Код, название и страна поставщика обязательны", "error")
        return _redirect("/suppliers")
    if email_clean and not _is_valid_email(email_clean):
        _set_flash(request, "Некорректный email поставщика", "error")
        return _redirect("/suppliers")
    db.add(
        Supplier(
            code=code_clean,
            name=name_clean,
            country=country_clean,
            contact_email=email_clean or None,
            is_active=active is not None,
        )
    )
    try:
        db.commit()
        _set_flash(request, "Поставщик создан", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "Код поставщика должен быть уникален", "error")
    return _redirect("/suppliers")


@app.post("/suppliers/{supplier_id}/update")
def update_supplier(
    supplier_id: int,
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    country: str = Form(...),
    contact_email: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    code_clean = code.strip()
    name_clean = name.strip()
    country_clean = country.strip()
    email_clean = contact_email.strip()
    if not code_clean or not name_clean or not country_clean:
        _set_flash(request, "Код, название и страна поставщика обязательны", "error")
        return _redirect("/suppliers")
    if email_clean and not _is_valid_email(email_clean):
        _set_flash(request, "Некорректный email поставщика", "error")
        return _redirect("/suppliers")
    supplier = db.get(Supplier, supplier_id)
    if supplier:
        new_is_active = active is not None
        if supplier.is_active and not new_is_active:
            usage = _supplier_usage_count(db, supplier.id)
            if usage:
                _set_flash(request, "Нельзя деактивировать поставщика: он используется в изделиях", "error")
                return _redirect("/suppliers")

        supplier.code = code_clean
        supplier.name = name_clean
        supplier.country = country_clean
        supplier.contact_email = email_clean or None
        supplier.is_active = new_is_active
        try:
            db.commit()
            _set_flash(request, "Поставщик обновлен", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код поставщика должен быть уникален", "error")
    return _redirect("/suppliers")


@app.post("/suppliers/{supplier_id}/delete")
def delete_supplier(
    supplier_id: int,
    request: Request,
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    supplier = db.get(Supplier, supplier_id)
    if not supplier:
        return _redirect("/suppliers")

    usage = _supplier_usage_count(db, supplier.id)
    if usage:
        _set_flash(request, f"Нельзя удалить поставщика: есть связанные изделия ({usage}).", "error")
        return _redirect(f"/suppliers/{supplier_id}")

    db.delete(supplier)
    db.commit()
    _set_flash(request, "Поставщик удален", "success")
    return _redirect("/suppliers")


@app.get("/dictionaries", response_class=HTMLResponse)
def dictionaries_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dictionaries = list(db.scalars(select(Dictionary).order_by(Dictionary.name.asc())).all())
    usage = {
        row[0]: row[1]
        for row in db.execute(
            select(Attribute.dictionary_id, func.count(Attribute.id))
            .where(Attribute.dictionary_id.is_not(None))
            .group_by(Attribute.dictionary_id)
        ).all()
    }
    return _render(
        request,
        "dictionaries/list.html",
        {
            "title": "Справочники",
            "dictionaries": dictionaries,
            "usage": usage,
            "can_manage": user.role in {"admin"},
            "user": user,
        },
    )


@app.post("/dictionaries")
def create_dictionary(
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    db.add(Dictionary(code=code.strip(), name=name.strip(), description=description.strip() or None))
    try:
        db.commit()
        _set_flash(request, "Справочник создан", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "Код справочника должен быть уникален", "error")
    return _redirect("/dictionaries")


@app.get("/dictionaries/{dictionary_id}", response_class=HTMLResponse)
def dictionary_detail(
    dictionary_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dictionary = db.get(Dictionary, dictionary_id)
    if not dictionary:
        _set_flash(request, "Справочник не найден", "error")
        return _redirect("/dictionaries")
    items = list(
        db.scalars(
            select(DictionaryItem)
            .where(DictionaryItem.dictionary_id == dictionary.id)
            .order_by(DictionaryItem.sort_order.asc(), DictionaryItem.label.asc())
        ).all()
    )
    used_by = list(db.scalars(select(Attribute).where(Attribute.dictionary_id == dictionary.id)).all())
    next_sort_order = _next_available_sort_order(db, dictionary.id)
    return _render(
        request,
        "dictionaries/detail.html",
        {
            "title": f"Справочник: {dictionary.name}",
            "dictionary": dictionary,
            "items": items,
            "used_by": used_by,
            "next_sort_order": next_sort_order,
            "can_manage": user.role in {"admin"},
            "user": user,
        },
    )


@app.post("/dictionaries/{dictionary_id}/update")
def update_dictionary(
    dictionary_id: int,
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    dictionary = db.get(Dictionary, dictionary_id)
    if dictionary:
        dictionary.code = code.strip()
        dictionary.name = name.strip()
        dictionary.description = description.strip() or None
        try:
            db.commit()
            _set_flash(request, "Справочник обновлен", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код справочника должен быть уникален", "error")
    return _redirect(f"/dictionaries/{dictionary_id}")


@app.post("/dictionaries/{dictionary_id}/delete")
def delete_dictionary(
    dictionary_id: int,
    request: Request,
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    dictionary = db.get(Dictionary, dictionary_id)
    if not dictionary:
        return _redirect("/dictionaries")

    used_by_attributes, used_in_categories, used_in_attribute_values = _dictionary_usage_counts(db, dictionary.id)
    if used_by_attributes or used_in_categories or used_in_attribute_values:
        reasons: list[str] = []
        if used_by_attributes:
            reasons.append(f"привязан к атрибутам ({used_by_attributes})")
        if used_in_categories:
            reasons.append(f"используется как категория в товарах ({used_in_categories})")
        if used_in_attribute_values:
            reasons.append(f"используется в значениях атрибутов товаров ({used_in_attribute_values})")
        _set_flash(request, f"Нельзя удалить справочник: {', '.join(reasons)}", "error")
        return _redirect(f"/dictionaries/{dictionary_id}")

    db.delete(dictionary)
    db.commit()
    _set_flash(request, "Справочник удален", "success")
    return _redirect("/dictionaries")


@app.post("/dictionaries/{dictionary_id}/items")
def create_dictionary_item(
    dictionary_id: int,
    request: Request,
    code: str = Form(...),
    label: str = Form(...),
    sort_order: int = Form(0),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    auto_sort_order = _next_available_sort_order(db, dictionary_id)
    db.add(
        DictionaryItem(
            dictionary_id=dictionary_id,
            code=code.strip(),
            label=label.strip(),
            sort_order=auto_sort_order,
            is_active=active is not None,
        )
    )
    try:
        db.commit()
        _set_flash(request, "Элемент справочника добавлен", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "Код элемента должен быть уникален внутри справочника", "error")
    return _redirect(f"/dictionaries/{dictionary_id}")


@app.post("/dictionary-items/{item_id}/update")
def update_dictionary_item(
    item_id: int,
    request: Request,
    code: str = Form(...),
    label: str = Form(...),
    sort_order: int = Form(0),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    item = db.get(DictionaryItem, item_id)
    if item:
        new_is_active = active is not None
        if item.is_active and not new_is_active:
            used_as_category, used_in_attribute_values = _dictionary_item_usage_counts(db, item.id)
            is_category_item = bool(
                db.scalar(select(Dictionary.id).where(Dictionary.id == item.dictionary_id, Dictionary.code == "category"))
            )
            if used_as_category and is_category_item:
                db.query(Product).filter(Product.category_item_id == item.id).update(
                    {Product.category_item_id: None},
                    synchronize_session=False,
                )
                used_as_category = 0
            if used_as_category or used_in_attribute_values:
                _set_flash(request, "Нельзя деактивировать элемент: он используется в товарах", "error")
                return _redirect(f"/dictionaries/{item.dictionary_id}")

        item.code = code.strip()
        item.label = label.strip()
        item.sort_order = sort_order
        item.is_active = new_is_active
        try:
            db.commit()
            _set_flash(request, "Элемент справочника обновлен", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код элемента должен быть уникален внутри справочника", "error")
        return _redirect(f"/dictionaries/{item.dictionary_id}")
    return _redirect("/dictionaries")


@app.post("/dictionary-items/{item_id}/delete")
def delete_dictionary_item(
    item_id: int,
    request: Request,
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    item = db.get(DictionaryItem, item_id)
    if not item:
        return _redirect("/dictionaries")

    used_as_category, used_in_attribute_values = _dictionary_item_usage_counts(db, item.id)
    is_category_item = bool(
        db.scalar(select(Dictionary.id).where(Dictionary.id == item.dictionary_id, Dictionary.code == "category"))
    )
    if used_as_category and is_category_item:
        db.query(Product).filter(Product.category_item_id == item.id).update(
            {Product.category_item_id: None},
            synchronize_session=False,
        )
        used_as_category = 0
    if used_as_category or used_in_attribute_values:
        reasons: list[str] = []
        if used_as_category:
            reasons.append(f"категория у товаров ({used_as_category})")
        if used_in_attribute_values:
            reasons.append(f"значения атрибутов товаров ({used_in_attribute_values})")
        _set_flash(request, f"Нельзя удалить элемент: используется в {', '.join(reasons)}", "error")
        return _redirect(f"/dictionaries/{item.dictionary_id}")

    dictionary_id = item.dictionary_id
    db.delete(item)
    db.commit()
    _set_flash(request, "Элемент справочника удален", "success")
    return _redirect(f"/dictionaries/{dictionary_id}")


@app.get("/attributes", response_class=HTMLResponse)
def attributes_page(
    request: Request,
    q: str = "",
    data_type: str = "",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt: Select[tuple[Attribute]] = select(Attribute).options(joinedload(Attribute.dictionary))
    if q:
        stmt = stmt.where(or_(Attribute.code.ilike(f"%{q}%"), Attribute.name.ilike(f"%{q}%")))
    if data_type:
        stmt = stmt.where(Attribute.data_type == data_type)

    attributes = list(db.scalars(stmt.order_by(Attribute.name.asc())).all())
    dictionaries = list(db.scalars(select(Dictionary).order_by(Dictionary.name.asc())).all())
    return _render(
        request,
        "attributes/list.html",
        {
            "title": "Атрибуты",
            "attributes": attributes,
            "dictionaries": dictionaries,
            "data_types": sorted(DATA_TYPES),
            "q": q,
            "selected_type": data_type,
            "can_manage": user.role in {"admin"},
            "user": user,
        },
    )


@app.post("/attributes")
def create_attribute(
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    data_type: str = Form(...),
    is_required: str | None = Form(None),
    is_multivalue: str | None = Form(None),
    dictionary_id_raw: str = Form(""),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    if data_type not in DATA_TYPES:
        _set_flash(request, "Некорректный тип данных", "error")
        return _redirect("/attributes")

    dictionary_id: int | None = None
    if data_type == "enum" and dictionary_id_raw.strip():
        try:
            dictionary_id = int(dictionary_id_raw)
        except ValueError:
            _set_flash(request, "Некорректный справочник", "error")
            return _redirect("/attributes")
        if not db.get(Dictionary, dictionary_id):
            _set_flash(request, "Справочник не найден", "error")
            return _redirect("/attributes")

    attribute = Attribute(
        code=code.strip(),
        name=name.strip(),
        data_type=data_type,
        is_required=is_required is not None,
        is_multivalue=is_multivalue is not None,
        dictionary_id=dictionary_id,
        is_active=True,
    )
    db.add(attribute)
    try:
        db.commit()
        assigned_count = _ensure_attribute_assigned_to_all_products(db, attribute.id)
        if assigned_count:
            db.commit()
        _set_flash(request, "Атрибут создан", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "Код атрибута должен быть уникален", "error")
    return _redirect("/attributes")


@app.get("/attributes/{attribute_id}", response_class=HTMLResponse)
def attribute_detail(
    attribute_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attribute = db.get(Attribute, attribute_id)
    if not attribute:
        _set_flash(request, "Атрибут не найден", "error")
        return _redirect("/attributes")
    dictionaries = list(db.scalars(select(Dictionary).order_by(Dictionary.name.asc())).all())
    return _render(
        request,
        "attributes/detail.html",
        {
            "title": f"Атрибут: {attribute.name}",
            "attribute": attribute,
            "dictionaries": dictionaries,
            "data_types": sorted(DATA_TYPES),
            "can_manage": user.role in {"admin"},
            "user": user,
        },
    )


@app.post("/attributes/{attribute_id}/update")
def update_attribute(
    attribute_id: int,
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    data_type: str = Form(...),
    is_required: str | None = Form(None),
    is_multivalue: str | None = Form(None),
    dictionary_id_raw: str = Form(""),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    attribute = db.get(Attribute, attribute_id)
    if not attribute:
        return _redirect("/attributes")

    if data_type not in DATA_TYPES:
        _set_flash(request, "Некорректный тип данных", "error")
        return _redirect(f"/attributes/{attribute_id}")

    if attribute.data_type != data_type and not can_change_attribute_type(db, attribute.id):
        _set_flash(request, "Нельзя изменить тип: по атрибуту уже есть значения", "error")
        return _redirect(f"/attributes/{attribute_id}")

    attribute.code = code.strip()
    attribute.name = name.strip()
    attribute.data_type = data_type
    attribute.is_required = is_required is not None
    attribute.is_multivalue = is_multivalue is not None
    dictionary_id: int | None = None
    if data_type == "enum" and dictionary_id_raw.strip():
        try:
            dictionary_id = int(dictionary_id_raw)
        except ValueError:
            _set_flash(request, "Некорректный справочник", "error")
            return _redirect(f"/attributes/{attribute_id}")
        if not db.get(Dictionary, dictionary_id):
            _set_flash(request, "Справочник не найден", "error")
            return _redirect(f"/attributes/{attribute_id}")
    attribute.dictionary_id = dictionary_id if data_type == "enum" else None

    try:
        db.commit()
        _set_flash(request, "Атрибут обновлен", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "Код атрибута должен быть уникален", "error")

    return _redirect(f"/attributes/{attribute_id}")


@app.post("/attributes/{attribute_id}/deactivate")
def deactivate_attribute(
    attribute_id: int,
    request: Request,
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    attribute = db.get(Attribute, attribute_id)
    if attribute:
        attribute.is_active = False
        db.commit()
        _set_flash(request, "Атрибут деактивирован", "success")
    return _redirect("/attributes")


@app.post("/attributes/{attribute_id}/delete")
def delete_attribute(
    attribute_id: int,
    request: Request,
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    attribute = db.get(Attribute, attribute_id)
    if not attribute:
        _set_flash(request, "Атрибут не найден", "error")
        return _redirect("/attributes")

    used_in_products = int(
        db.scalar(
            select(func.count(ProductAttributeAssignment.id)).where(ProductAttributeAssignment.attribute_id == attribute_id)
        )
        or 0
    )
    if used_in_products:
        _set_flash(
            request,
            f"Нельзя удалить атрибут: он используется в изделиях ({used_in_products}). Сначала уберите его из карточек.",
            "error",
        )
        return _redirect(f"/attributes/{attribute_id}")

    db.delete(attribute)
    db.commit()
    _set_flash(request, "Атрибут удален", "success")
    return _redirect("/attributes")


@app.get("/products", response_class=HTMLResponse)
def products_page(
    request: Request,
    q: str = "",
    full_search: str = "",
    status_filter: str = "",
    category_id_raw: str = "",
    collection_id_raw: str = "",
    sort: str = "updated_desc",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    category_id: int | None = None
    collection_id: int | None = None
    if category_id_raw.strip():
        try:
            category_id = int(category_id_raw)
        except ValueError:
            category_id = None
    if collection_id_raw.strip():
        try:
            collection_id = int(collection_id_raw)
        except ValueError:
            collection_id = None

    stmt: Select[tuple[Product]] = select(Product).options(
        joinedload(Product.category_item),
        selectinload(Product.spec).joinedload(ProductSpec.collection),
        selectinload(Product.spec).joinedload(ProductSpec.supplier),
    )
    if status_filter:
        stmt = stmt.where(Product.status == status_filter)
    if category_id:
        stmt = stmt.where(Product.category_item_id == category_id)
    if collection_id:
        stmt = stmt.join(ProductSpec, ProductSpec.product_id == Product.id).where(ProductSpec.collection_id == collection_id)

    if sort == "name_asc":
        stmt = stmt.order_by(Product.name.asc())
    elif sort == "created_desc":
        stmt = stmt.order_by(Product.created_at.desc())
    else:
        stmt = stmt.order_by(Product.updated_at.desc())

    products = list(db.scalars(stmt).all())
    if q:
        q_norm = q.casefold().strip()
        products = [p for p in products if q_norm in (p.name or "").casefold() or q_norm in (p.sku or "").casefold()]

    full_search_attributes = list(
        db.scalars(
            select(Attribute)
            .options(joinedload(Attribute.dictionary))
            .where(Attribute.is_active.is_(True))
            .order_by(Attribute.name.asc())
        ).all()
    )
    full_search_values: dict[int, dict[str, object]] = {}
    full_search_enum_options: dict[int, list[DictionaryItem]] = {}
    full_search_active = False
    query = request.query_params
    for attr in full_search_attributes:
        if attr.data_type == "bool":
            raw = (query.get(f"fs_a_{attr.id}_bool") or "").strip()
            full_search_values[attr.id] = {"mode": "bool", "value": raw}
            if raw in {"0", "1"}:
                full_search_active = True
        elif attr.data_type == "enum":
            key = f"fs_a_{attr.id}_enum"
            if attr.is_multivalue:
                selected = [int(v.strip()) for v in query.getlist(key) if v.strip().isdigit()]
            else:
                selected_one = (query.get(key) or "").strip()
                selected = [int(selected_one)] if selected_one.isdigit() else []
            full_search_values[attr.id] = {"mode": "enum", "values": selected}
            if selected:
                full_search_active = True
            if attr.dictionary_id:
                full_search_enum_options[attr.id] = list(
                    db.scalars(
                        select(DictionaryItem)
                        .where(
                            DictionaryItem.dictionary_id == attr.dictionary_id,
                            DictionaryItem.is_active.is_(True),
                        )
                        .order_by(DictionaryItem.sort_order.asc(), DictionaryItem.label.asc())
                    ).all()
                )
            else:
                full_search_enum_options[attr.id] = []
        else:
            raw = (query.get(f"fs_a_{attr.id}_value") or "").strip()
            full_search_values[attr.id] = {"mode": "text", "value": raw}
            if raw:
                full_search_active = True

    if full_search_active and products:
        product_ids = [p.id for p in products]
        assignments_by_product: dict[int, list[ProductAttributeAssignment]] = {}
        assignments = list(
            db.scalars(
                select(ProductAttributeAssignment)
                .options(
                    joinedload(ProductAttributeAssignment.attribute),
                    selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
                )
                .where(ProductAttributeAssignment.product_id.in_(product_ids))
            ).all()
        )
        for assignment in assignments:
            assignments_by_product.setdefault(assignment.product_id, []).append(assignment)
        products = [
            p
            for p in products
            if _product_matches_attribute_filters(
                p,
                assignments_by_product.get(p.id, []),
                full_search_attributes,
                full_search_values,
            )
        ]
    product_task_counts: dict[int, int] = {}
    if products:
        product_ids = [p.id for p in products]
        rows = db.execute(
            select(Task.product_id, func.count(Task.id))
            .where(Task.product_id.in_(product_ids))
            .group_by(Task.product_id)
        ).all()
        product_task_counts = {int(product_id): int(count) for product_id, count in rows if product_id is not None}
    category_items = get_category_items(db)
    collections = list(db.scalars(select(Collection).where(Collection.is_active.is_(True)).order_by(Collection.year.desc())).all())

    return _render(
        request,
        "products/list.html",
        {
            "title": "Изделия",
            "products": products,
            "statuses": sorted(PRODUCT_STATUSES),
            "q": q,
            "full_search": full_search,
            "full_search_enabled": full_search_active,
            "full_search_active": full_search_active,
            "full_search_attributes": full_search_attributes,
            "full_search_values": full_search_values,
            "full_search_enum_options": full_search_enum_options,
            "status_filter": status_filter,
            "category_items": category_items,
            "selected_category_id": category_id,
            "collections": collections,
            "product_collection_map": _product_collection_map(db),
            "selected_collection_id": collection_id,
            "sort": sort,
            "product_task_counts": product_task_counts,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.get("/tasks/by-product/{product_id}", response_class=HTMLResponse)
def tasks_by_product_page(
    product_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        _set_flash(request, "Изделие не найдено", "error")
        return _redirect("/products")

    tasks = list(
        db.scalars(
            select(Task)
            .options(
                joinedload(Task.author),
                joinedload(Task.assignee),
                joinedload(Task.queue),
                joinedload(Task.collection),
            )
            .where(Task.product_id == product_id)
            .order_by(Task.created_at.desc())
        ).all()
    )
    return _render(
        request,
        "tasks/by_product.html",
        {
            "title": f"Задачи по изделию: {product.sku}",
            "product": product,
            "tasks": tasks,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/products")
def create_product(
    request: Request,
    sku: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    status: str = Form("draft"),
    category_id_raw: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    sku_clean = sku.strip()
    name_clean = name.strip()
    if not sku_clean or not name_clean:
        _set_flash(request, "SKU и наименование изделия обязательны", "error")
        return _redirect("/products")
    if status not in PRODUCT_STATUSES:
        _set_flash(request, "Некорректный статус изделия", "error")
        return _redirect("/products")

    category_id: int | None = None
    if category_id_raw.strip():
        try:
            category_id = int(category_id_raw)
        except ValueError:
            _set_flash(request, "Некорректная категория", "error")
            return _redirect("/products")
        if not _is_valid_category_item(db, category_id):
            _set_flash(request, "Выбрана несуществующая или неактивная категория", "error")
            return _redirect("/products")

    product = Product(
        sku=sku_clean,
        name=name_clean,
        description=description.strip() or None,
        status=status,
        category_item_id=category_id,
        created_by=user.id,
        updated_by=user.id,
    )
    db.add(product)
    try:
        db.commit()
        if product.status == "active":
            errors = validate_product_completeness(db, product)
            if errors:
                product.status = "draft"
                db.commit()
                _set_flash(
                    request,
                    "Изделие создано как draft: для активации заполните обязательные атрибуты",
                    "error",
                )
                return _redirect(f"/products/{product.id}")
        _set_flash(request, "Изделие создано", "success")
        return _redirect(f"/products/{product.id}")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "SKU должен быть уникален", "error")
        return _redirect("/products")


@app.get("/products/{product_id}", response_class=HTMLResponse)
def product_detail(
    product_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = db.scalar(
        select(Product)
        .options(
            joinedload(Product.category_item),
            joinedload(Product.designer),
            joinedload(Product.product_manager),
            joinedload(Product.pattern_maker),
            joinedload(Product.technologist),
            joinedload(Product.department_head),
            selectinload(Product.spec).joinedload(ProductSpec.collection),
            selectinload(Product.spec).joinedload(ProductSpec.supplier),
            selectinload(Product.files).joinedload(ProductFile.uploader),
        )
        .where(Product.id == product_id)
    )
    if not product:
        _set_flash(request, "Изделие не найдено", "error")
        return _redirect("/products")

    category_items = get_category_items(db)
    collections = list(db.scalars(select(Collection).where(Collection.is_active.is_(True)).order_by(Collection.year.desc())).all())
    suppliers = list(db.scalars(select(Supplier).where(Supplier.is_active.is_(True)).order_by(Supplier.name.asc())).all())
    spec_dictionary_options = _get_spec_dictionary_options(db)
    active_users = list(db.scalars(select(User).where(User.is_active.is_(True)).order_by(User.login.asc())).all())
    attribute_assignments = _load_product_attribute_assignments(db, product_id)
    enum_options_by_attr: dict[int, list[DictionaryItem]] = {}
    for assignment in attribute_assignments:
        attr = assignment.attribute
        if not attr or attr.data_type != "enum" or not attr.dictionary_id:
            continue
        enum_options_by_attr[assignment.id] = list(
            db.scalars(
                select(DictionaryItem)
                .where(
                    DictionaryItem.dictionary_id == attr.dictionary_id,
                    DictionaryItem.is_active.is_(True),
                )
                .order_by(DictionaryItem.sort_order.asc(), DictionaryItem.label.asc())
            ).all()
        )
    product_tasks = list(
        db.scalars(
            select(Task)
            .options(
                joinedload(Task.author),
                joinedload(Task.assignee),
                joinedload(Task.queue),
            )
            .where(Task.product_id == product_id)
            .order_by(Task.created_at.desc())
        ).all()
    )
    return _render(
        request,
        "products/detail.html",
        {
            "title": f"Изделие: {product.name}",
            "product": product,
            "statuses": sorted(PRODUCT_STATUSES),
            "category_items": category_items,
            "collections": collections,
            "product_collection_map": _product_collection_map(db),
            "suppliers": suppliers,
            "spec_dictionary_options": spec_dictionary_options,
            "sample_stages": SAMPLE_STAGES,
            "file_categories": FILE_CATEGORIES,
            "active_users": active_users,
            "attribute_assignments": attribute_assignments,
            "enum_options_by_attr": enum_options_by_attr,
            "product_tasks": product_tasks,
            "get_value_view": get_value_view,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.get("/products/{product_id}/pdf")
def product_pdf(
    product_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = db.scalar(
        select(Product)
        .options(
            joinedload(Product.category_item),
            joinedload(Product.designer),
            joinedload(Product.product_manager),
            joinedload(Product.pattern_maker),
            joinedload(Product.technologist),
            joinedload(Product.department_head),
            selectinload(Product.spec).joinedload(ProductSpec.collection),
            selectinload(Product.spec).joinedload(ProductSpec.supplier),
            selectinload(Product.files).joinedload(ProductFile.uploader),
        )
        .where(Product.id == product_id)
    )
    if not product:
        return Response(content="", status_code=404)

    attribute_assignments = _load_product_attribute_assignments(db, product_id)
    product_tasks = list(
        db.scalars(
            select(Task)
            .options(
                joinedload(Task.author),
                joinedload(Task.assignee),
                joinedload(Task.queue),
            )
            .where(Task.product_id == product_id)
            .order_by(Task.created_at.desc())
        ).all()
    )
    pdf_data = _product_pdf_bytes(product, attribute_assignments, product_tasks)
    safe_sku = re.sub(r"[^A-Za-z0-9._-]+", "_", (product.sku or f"product_{product.id}"))
    filename = f"{safe_sku}_card.pdf"
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/products/{product_id}/update")
def update_product(
    product_id: int,
    request: Request,
    sku: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    status: str = Form("draft"),
    category_id_raw: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    sku_clean = sku.strip()
    name_clean = name.strip()
    if not sku_clean or not name_clean:
        _set_flash(request, "SKU и наименование изделия обязательны", "error")
        return _redirect(f"/products/{product_id}")

    if status not in PRODUCT_STATUSES:
        _set_flash(request, "Некорректный статус изделия", "error")
        return _redirect(f"/products/{product_id}")

    category_id: int | None = None
    if category_id_raw.strip():
        try:
            category_id = int(category_id_raw)
        except ValueError:
            _set_flash(request, "Некорректная категория", "error")
            return _redirect(f"/products/{product_id}")
        if not _is_valid_category_item(db, category_id):
            _set_flash(request, "Выбрана несуществующая или неактивная категория", "error")
            return _redirect(f"/products/{product_id}")

    product.sku = sku_clean
    product.name = name_clean
    product.description = description.strip() or None
    product.status = status
    product.category_item_id = category_id
    product.updated_at = datetime.utcnow()
    product.updated_by = user.id

    if product.status == "active":
        errors = validate_product_completeness(db, product)
        if errors:
            db.rollback()
            _set_flash(request, "Нельзя активировать изделие: заполните обязательные атрибуты", "error")
            return _redirect(f"/products/{product_id}")

    try:
        db.commit()
        _set_flash(request, "Изделие обновлено", "success")
    except IntegrityError:
        db.rollback()
        _set_flash(request, "SKU должен быть уникален", "error")

    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/cover")
def upload_product_cover(
    product_id: int,
    request: Request,
    cover_file: UploadFile = File(...),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")

    allowed = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
    }
    ext = allowed.get(cover_file.content_type or "")
    if not ext:
        _set_flash(request, "Допустимые форматы: PNG, JPG, WEBP, SVG", "error")
        return _redirect(f"/products/{product_id}")

    _ensure_uploads_dir()
    file_name = f"product_{product_id}_{uuid4().hex[:12]}{ext}"
    dst = UPLOAD_DIR / file_name
    data = cover_file.file.read()
    if not data:
        _set_flash(request, "Пустой файл", "error")
        return _redirect(f"/products/{product_id}")
    dst.write_bytes(data)

    if product.cover_image_path and product.cover_image_path.startswith("/static/uploads/"):
        old_name = product.cover_image_path.split("/static/uploads/", 1)[1]
        old_path = UPLOAD_DIR / old_name
        if old_path.exists():
            old_path.unlink()

    product.cover_image_path = f"/static/uploads/{file_name}"
    product.updated_by = user.id
    product.updated_at = datetime.utcnow()
    db.commit()
    _set_flash(request, "Обложка загружена", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/files")
def upload_product_file(
    product_id: int,
    request: Request,
    category: str = Form(...),
    title: str = Form(""),
    product_file: UploadFile = File(...),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    if category not in FILE_CATEGORIES:
        _set_flash(request, "Некорректная категория файла", "error")
        return _redirect(f"/products/{product_id}")

    suffix = Path(product_file.filename or "file.bin").suffix.lower()
    allowed_suffixes = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".csv",
        ".txt",
        ".zip",
        ".rar",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".svg",
        ".dxf",
        ".plt",
    }
    if suffix and suffix not in allowed_suffixes:
        _set_flash(request, "Недопустимый тип файла", "error")
        return _redirect(f"/products/{product_id}")

    _ensure_uploads_dir()
    file_name = f"pf_{product_id}_{uuid4().hex[:12]}{suffix or '.bin'}"
    dst = PRODUCT_FILES_DIR / file_name
    data = product_file.file.read()
    if not data:
        _set_flash(request, "Пустой файл", "error")
        return _redirect(f"/products/{product_id}")
    dst.write_bytes(data)

    rec = ProductFile(
        product_id=product_id,
        category=category,
        title=title.strip() or None,
        original_name=product_file.filename or file_name,
        file_path=f"/static/product_files/{file_name}",
        mime_type=product_file.content_type,
        uploaded_by=user.id,
    )
    db.add(rec)
    product.updated_by = user.id
    product.updated_at = datetime.utcnow()
    db.commit()
    _set_flash(request, "Файл прикреплен к карточке", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/files/{file_id}/delete")
def delete_product_file(
    product_id: int,
    file_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    rec = db.scalar(select(ProductFile).where(ProductFile.id == file_id, ProductFile.product_id == product_id))
    if not rec:
        return _redirect(f"/products/{product_id}")

    if rec.file_path.startswith("/static/product_files/"):
        file_name = rec.file_path.split("/static/product_files/", 1)[1]
        file_path = PRODUCT_FILES_DIR / file_name
        if file_path.exists():
            file_path.unlink()

    product = db.get(Product, product_id)
    db.delete(rec)
    if product:
        product.updated_by = user.id
        product.updated_at = datetime.utcnow()
    db.commit()
    _set_flash(request, "Файл удален", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/spec")
async def update_product_spec(
    product_id: int,
    request: Request,
    collection_id_raw: str = Form(""),
    supplier_id_raw: str = Form(""),
    style_type: str = Form(""),
    capsule: str = Form(""),
    silhouette: str = Form(""),
    fit_type: str = Form(""),
    length_cm_raw: str = Form(""),
    shell_material: str = Form(""),
    lining_material: str = Form(""),
    insulation: str = Form(""),
    sample_stage: str = Form(""),
    planned_cost_raw: str = Form(""),
    actual_cost_raw: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    form = await request.form()

    spec_dictionary_options = _get_spec_dictionary_options(db)

    def parse_int(raw: str) -> int | None:
        return int(raw) if raw.strip() else None

    def parse_float(raw: str) -> float | None:
        return float(raw.replace(",", ".")) if raw.strip() else None

    try:
        collection_id = parse_int(collection_id_raw)
        supplier_id = parse_int(supplier_id_raw)
        length_cm = parse_float(length_cm_raw)
        planned_cost = parse_float(planned_cost_raw)
        actual_cost = parse_float(actual_cost_raw)
    except ValueError:
        _set_flash(request, "Проверьте числовые поля fashion-спецификации", "error")
        return _redirect(f"/products/{product_id}")

    if collection_id is not None:
        collection = db.get(Collection, collection_id)
        if not collection or not collection.is_active:
            _set_flash(request, "Выбрана несуществующая или неактивная коллекция", "error")
            return _redirect(f"/products/{product_id}")
    if supplier_id is not None:
        supplier = db.get(Supplier, supplier_id)
        if not supplier or not supplier.is_active:
            _set_flash(request, "Выбран несуществующий или неактивный поставщик", "error")
            return _redirect(f"/products/{product_id}")
    if sample_stage.strip() and sample_stage.strip() not in SAMPLE_STAGES:
        _set_flash(request, "Некорректный этап образца", "error")
        return _redirect(f"/products/{product_id}")

    spec = db.scalar(select(ProductSpec).where(ProductSpec.product_id == product_id))
    if not spec:
        spec = ProductSpec(product_id=product_id)
        db.add(spec)

    def normalize_spec_value(field_name: str, raw: str) -> str | None:
        value = raw.strip()
        if not value:
            return None
        options = spec_dictionary_options.get(field_name, [])
        if not options:
            return value
        allowed = {item.label for item in options}
        if value not in allowed:
            raise ValueError(field_name)
        return value

    try:
        style_type_value = normalize_spec_value("style_type", style_type)
        capsule_value = normalize_spec_value("capsule", capsule)
        silhouette_value = normalize_spec_value("silhouette", silhouette)
        fit_type_value = normalize_spec_value("fit_type", fit_type)
        shell_material_value = normalize_spec_value("shell_material", shell_material)
        lining_material_value = normalize_spec_value("lining_material", lining_material)
        insulation_value = normalize_spec_value("insulation", insulation)
    except ValueError:
        _set_flash(request, "Выберите значения из справочника. Ручной ввод для этих полей отключен.", "error")
        return _redirect(f"/products/{product_id}")

    spec.collection_id = collection_id
    spec.supplier_id = supplier_id
    spec.style_type = style_type_value
    spec.capsule = capsule_value
    spec.silhouette = silhouette_value
    spec.fit_type = fit_type_value
    spec.length_cm = length_cm
    spec.shell_material = shell_material_value
    spec.lining_material = lining_material_value
    spec.insulation = insulation_value
    spec.sample_stage = sample_stage.strip() or None
    spec.planned_cost = planned_cost
    spec.actual_cost = actual_cost

    assignment_errors: list[str] = []
    assignments = _load_product_attribute_assignments(db, product_id)
    for assignment in assignments:
        attr = assignment.attribute
        if not attr:
            continue
        field_prefix = f"attr_{assignment.id}"
        if attr.data_type == "bool":
            payload: str | bool | list[str] | None = form.get(f"{field_prefix}_bool") is not None
        elif attr.data_type == "enum":
            if attr.is_multivalue:
                payload = [str(v) for v in form.getlist(f"{field_prefix}_enum") if str(v).strip()]
            else:
                payload = str(form.get(f"{field_prefix}_enum", ""))
        else:
            payload = str(form.get(f"{field_prefix}_value", ""))
        assignment_errors.extend(validate_and_set_values(db, assignment, attr, payload))

    if assignment_errors:
        db.rollback()
        _set_flash(request, assignment_errors[0], "error")
        return _redirect(f"/products/{product_id}")

    product.updated_by = user.id
    product.updated_at = datetime.utcnow()
    db.commit()
    _set_flash(request, "Fashion-спецификация обновлена", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/team")
def update_product_team(
    product_id: int,
    request: Request,
    designer_id_raw: str = Form(""),
    product_manager_id_raw: str = Form(""),
    pattern_maker_id_raw: str = Form(""),
    technologist_id_raw: str = Form(""),
    department_head_id_raw: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")

    def parse_user_id(raw: str) -> int | None:
        return int(raw) if raw.strip() else None

    try:
        designer_id = parse_user_id(designer_id_raw)
        product_manager_id = parse_user_id(product_manager_id_raw)
        pattern_maker_id = parse_user_id(pattern_maker_id_raw)
        technologist_id = parse_user_id(technologist_id_raw)
        department_head_id = parse_user_id(department_head_id_raw)
    except ValueError:
        _set_flash(request, "Проверьте корректность выбранных пользователей", "error")
        return _redirect(f"/products/{product_id}")

    def ensure_active(user_id: int | None) -> bool:
        if user_id is None:
            return True
        return db.scalar(select(User.id).where(User.id == user_id, User.is_active.is_(True))) is not None

    if not all(
        ensure_active(uid)
        for uid in [designer_id, product_manager_id, pattern_maker_id, technologist_id, department_head_id]
    ):
        _set_flash(request, "Выбран несуществующий или неактивный пользователь", "error")
        return _redirect(f"/products/{product_id}")

    product.designer_id = designer_id
    product.product_manager_id = product_manager_id
    product.pattern_maker_id = pattern_maker_id
    product.technologist_id = technologist_id
    product.department_head_id = department_head_id
    product.updated_by = user.id
    product.updated_at = datetime.utcnow()
    db.commit()
    _set_flash(request, "Команда изделия обновлена", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/archive")
def archive_product(
    product_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if product:
        product.status = "archived"
        product.updated_by = user.id
        product.updated_at = datetime.utcnow()
        db.commit()
        _set_flash(request, "Изделие архивировано", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/unarchive")
def unarchive_product(
    product_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")

    completeness_errors = validate_product_completeness(db, product)
    product.status = "active" if not completeness_errors else "draft"
    product.updated_by = user.id
    product.updated_at = datetime.utcnow()
    db.commit()
    if product.status == "active":
        _set_flash(request, "Изделие убрано из архива и активировано", "success")
    else:
        _set_flash(request, "Изделие убрано из архива и переведено в черновик", "success")
    return _redirect(f"/products/{product_id}")


@app.get("/products/{product_id}/attributes", response_class=HTMLResponse)
def product_attributes_page(
    product_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        _set_flash(request, "Изделие не найдено", "error")
        return _redirect("/products")

    assignments = _load_product_attribute_assignments(db, product_id)
    assigned_ids = {a.attribute_id for a in assignments}
    available_attributes = list(
        db.scalars(
            select(Attribute)
            .where(Attribute.is_active.is_(True), Attribute.id.not_in(assigned_ids) if assigned_ids else True)
            .order_by(Attribute.name.asc())
        ).all()
    )

    return _render(
        request,
        "products/attributes.html",
        {
            "title": f"Атрибуты изделия: {product.name}",
            "product": product,
            "assignments": assignments,
            "available_attributes": available_attributes,
            "get_value_view": get_value_view,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/products/{product_id}/attributes/add")
def add_product_attribute(
    product_id: int,
    request: Request,
    attribute_id: int = Form(...),
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    if not db.get(Product, product_id):
        _set_flash(request, "Изделие не найдено", "error")
        return _redirect("/products")
    attribute = db.scalar(select(Attribute).where(Attribute.id == attribute_id, Attribute.is_active.is_(True)))
    if not attribute:
        _set_flash(request, "Некорректный или неактивный атрибут", "error")
        return _redirect(f"/products/{product_id}/attributes")

    exists = db.execute(
        select(ProductAttributeAssignment.id).where(
            ProductAttributeAssignment.product_id == product_id,
            ProductAttributeAssignment.attribute_id == attribute_id,
        )
    ).first()
    if exists:
        _set_flash(request, "Атрибут уже назначен изделию", "error")
        return _redirect(f"/products/{product_id}/attributes")

    assignment = ProductAttributeAssignment(product_id=product_id, attribute_id=attribute_id)
    db.add(assignment)
    db.commit()
    _set_flash(request, "Атрибут назначен. Заполните значение.", "success")
    return _redirect(f"/products/{product_id}/attributes/{assignment.id}")


@app.get("/products/{product_id}/attributes/{assignment_id}", response_class=HTMLResponse)
def edit_product_attribute(
    product_id: int,
    assignment_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assignment = db.scalar(
        select(ProductAttributeAssignment)
        .options(
            joinedload(ProductAttributeAssignment.attribute),
            selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
        )
        .where(ProductAttributeAssignment.id == assignment_id, ProductAttributeAssignment.product_id == product_id)
    )
    if not assignment:
        _set_flash(request, "Назначение атрибута не найдено", "error")
        return _redirect(f"/products/{product_id}/attributes")

    enum_items: list[DictionaryItem] = []
    selected_enum_ids: list[int] = []
    if assignment.attribute.data_type == "enum" and assignment.attribute.dictionary_id:
        enum_items = list(
            db.scalars(
                select(DictionaryItem)
                .where(
                    DictionaryItem.dictionary_id == assignment.attribute.dictionary_id,
                    DictionaryItem.is_active.is_(True),
                )
                .order_by(DictionaryItem.sort_order.asc(), DictionaryItem.label.asc())
            ).all()
        )
        selected_enum_ids = [v.dictionary_item_id for v in assignment.values if v.dictionary_item_id is not None]

    return _render(
        request,
        "products/attribute_edit.html",
        {
            "title": f"Значение атрибута: {assignment.attribute.name}",
            "assignment": assignment,
            "enum_items": enum_items,
            "selected_enum_ids": selected_enum_ids,
            "get_value_view": get_value_view,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/products/{product_id}/attributes/{assignment_id}/update")
def update_product_attribute(
    product_id: int,
    assignment_id: int,
    request: Request,
    value_text: str = Form(""),
    value_bool: str | None = Form(None),
    enum_values: list[str] = Form(default=[]),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    assignment = db.scalar(
        select(ProductAttributeAssignment)
        .options(joinedload(ProductAttributeAssignment.attribute), selectinload(ProductAttributeAssignment.values))
        .where(ProductAttributeAssignment.id == assignment_id, ProductAttributeAssignment.product_id == product_id)
    )
    if not assignment:
        return _redirect(f"/products/{product_id}/attributes")

    attr = assignment.attribute
    payload: str | bool | list[str] | None
    if attr.data_type == "bool":
        payload = value_bool is not None
    elif attr.data_type == "enum":
        payload = enum_values
    else:
        payload = value_text

    errors = validate_and_set_values(db, assignment, attr, payload)
    if errors:
        db.rollback()
        _set_flash(request, "; ".join(errors), "error")
        return _redirect(f"/products/{product_id}/attributes/{assignment_id}")

    product = db.get(Product, product_id)
    if product:
        product.updated_by = user.id
        product.updated_at = datetime.utcnow()
    db.commit()
    _set_flash(request, "Значение атрибута обновлено", "success")
    return _redirect(f"/products/{product_id}/attributes")


@app.post("/products/{product_id}/attributes/{assignment_id}/remove")
def remove_product_attribute(
    product_id: int,
    assignment_id: int,
    request: Request,
    _: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    assignment = db.scalar(
        select(ProductAttributeAssignment)
        .options(joinedload(ProductAttributeAssignment.attribute))
        .where(
            ProductAttributeAssignment.id == assignment_id,
            ProductAttributeAssignment.product_id == product_id,
        )
    )
    if assignment:
        if assignment.attribute and assignment.attribute.is_required:
            _set_flash(request, "Нельзя снять обязательный атрибут с изделия", "error")
            return _redirect(f"/products/{product_id}/attributes")
        db.delete(assignment)
        db.commit()
        _set_flash(request, "Атрибут снят с изделия", "success")
    return _redirect(f"/products/{product_id}/attributes")





