from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, RedirectResponse, Response
import qrcode
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
    "content-manager": "Контент-менеджер",
    "dictionary-manager": "Менеджер справочников",
    "read-only": "Только просмотр",
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


def _render(request: Request, template_name: str, context: dict) -> HTMLResponse:
    context["request"] = request
    context["flash"] = _get_flash(request)
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
    _ensure_uploads_dir()
    db = SessionLocal()
    try:
        if not db.scalar(select(User).where(User.login == "admin")):
            demo_users = [
                User(login="admin", password_hash=hash_password("admin"), role="admin", is_active=True),
                User(login="content", password_hash=hash_password("content"), role="content-manager", is_active=True),
                User(login="dict", password_hash=hash_password("dict"), role="dictionary-manager", is_active=True),
                User(login="viewer", password_hash=hash_password("viewer"), role="read-only", is_active=True),
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
    return _render(request, "login.html", {"title": "Вход"})


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
    if role not in ROLES:
        _set_flash(request, "Некорректная роль", "error")
        return _redirect("/users")
    department_dict, _ = _get_dictionary_items(db, "department")
    position_dict, _ = _get_dictionary_items(db, "position")

    def parse_item_id(raw: str, dictionary: Dictionary | None) -> int | None:
        if not raw.strip():
            return None
        if not dictionary:
            raise ValueError("missing-dictionary")
        item_id = int(raw)
        item_exists = db.scalar(
            select(DictionaryItem.id).where(
                DictionaryItem.id == item_id,
                DictionaryItem.dictionary_id == dictionary.id,
                DictionaryItem.is_active.is_(True),
            )
        )
        if not item_exists:
            raise ValueError("bad-item")
        return item_id

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
            login=login.strip(),
            password_hash=hash_password(password),
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
    position_id_raw: str = Form(""),
    _: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if not target:
        _set_flash(request, "Пользователь не найден", "error")
        return _redirect("/users")
    department_dict, _ = _get_dictionary_items(db, "department")
    position_dict, _ = _get_dictionary_items(db, "position")

    def parse_item_id(raw: str, dictionary: Dictionary | None) -> int | None:
        if not raw.strip():
            return None
        if not dictionary:
            raise ValueError("missing-dictionary")
        item_id = int(raw)
        item_exists = db.scalar(
            select(DictionaryItem.id).where(
                DictionaryItem.id == item_id,
                DictionaryItem.dictionary_id == dictionary.id,
                DictionaryItem.is_active.is_(True),
            )
        )
        if not item_exists:
            raise ValueError("bad-item")
        return item_id

    try:
        department_item_id = parse_item_id(department_id_raw, department_dict)
        position_item_id = parse_item_id(position_id_raw, position_dict)
    except ValueError:
        _set_flash(request, "Некорректное подразделение или должность", "error")
        return _redirect(f"/users/{user_id}")

    department_label = db.scalar(select(DictionaryItem.label).where(DictionaryItem.id == department_item_id)) if department_item_id else None
    position_label = db.scalar(select(DictionaryItem.label).where(DictionaryItem.id == position_item_id)) if position_item_id else None
    target.full_name = full_name.strip() or None
    target.department = department_label
    target.position = position_label
    target.department_item_id = department_item_id
    target.position_item_id = position_item_id
    db.commit()
    _set_flash(request, "Профиль пользователя обновлен", "success")
    return _redirect(f"/users/{user_id}")


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
            "can_manage": user.role in {"admin", "content-manager"},
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
            "status_order": TASK_STATUS_ORDER,
            "priorities": TASK_PRIORITIES,
            "q": q,
            "selected_status": status,
            "selected_priority": priority,
            "selected_assignee_id": assignee_id,
            "selected_collection_id": collection_id,
            "selected_product_id": product_id,
            "can_manage": user.role in {"admin", "content-manager"},
            "user": user,
        },
    )


@app.post("/queues")
def create_queue(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    active: str | None = Form(None),
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    created = False
    for _ in range(5):
        existing_codes = list(db.scalars(select(TaskQueue.code)).all())
        next_code = _next_auto_code(existing_codes, "Q")
        db.add(
            TaskQueue(
                code=next_code,
                name=name.strip(),
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    queue = db.get(TaskQueue, queue_id)
    if queue:
        queue.name = name.strip()
        queue.description = description.strip() or None
        queue.is_active = active is not None
        try:
            db.commit()
            _set_flash(request, "Очередь обновлена", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код очереди должен быть уникален", "error")
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
            "can_manage": user.role in {"admin", "content-manager"},
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
            "status_order": TASK_STATUS_ORDER,
            "priorities": TASK_PRIORITIES,
            "q": q,
            "selected_status": status,
            "selected_priority": priority,
            "selected_queue_id": queue_id,
            "selected_assignee_id": assignee_id,
            "selected_collection_id": collection_id,
            "selected_product_id": product_id,
            "can_manage": user.role in {"admin", "content-manager"},
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    created = False
    filter_queue_id = int(filter_queue_id_raw) if filter_queue_id_raw.strip().isdigit() else None
    if filter_queue_id is not None and not db.get(TaskQueue, filter_queue_id):
        _set_flash(request, "Некорректная очередь для фильтра доски", "error")
        return _redirect("/boards")
    for _ in range(5):
        existing_codes = list(db.scalars(select(TaskBoard.code)).all())
        next_code = _next_auto_code(existing_codes, "K")
        db.add(
            TaskBoard(
                code=next_code,
                name=name.strip(),
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    board = db.scalar(select(TaskBoard).options(joinedload(TaskBoard.filter_queue)).where(TaskBoard.id == board_id))
    if board:
        filter_queue_id = int(filter_queue_id_raw) if filter_queue_id_raw.strip().isdigit() else None
        if filter_queue_id is not None and not db.get(TaskQueue, filter_queue_id):
            _set_flash(request, "Некорректная очередь для фильтра доски", "error")
            return _redirect(f"/boards/{board_id}")
        board.name = name.strip()
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
            "q": q,
            "selected_status": status_filter,
            "selected_priority": priority,
            "selected_queue_id": queue_id,
            "selected_assignee_id": assignee_id,
            "selected_collection_id": collection_id,
            "selected_product_id": product_id,
            "can_manage": user.role in {"admin", "content-manager"},
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
    user: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
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
        task = Task(
            title=title.strip(),
            comment=comment.strip() or None,
            status=status,
            priority=priority,
            tags=tags.strip() or None,
            start_date=parse_date(start_date),
            end_date=parse_date(end_date),
            deadline=parse_date(deadline),
            author_id=user.id,
            assignee_id=parse_int(assignee_id_raw),
            queue_id=parse_int(queue_id_raw),
            board_id=None,
            collection_id=parse_int(collection_id_raw),
            product_id=parse_int(product_id_raw),
        )
    except ValueError:
        _set_flash(request, "Проверьте даты задачи (формат YYYY-MM-DD)", "error")
        return _redirect(request.headers.get("referer") or "/queues")

    if task.queue_id is None:
        _set_flash(request, "Задача должна принадлежать очереди", "error")
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
            "status_order": TASK_STATUS_ORDER,
            "priorities": TASK_PRIORITIES,
            "can_manage": user.role in {"admin", "content-manager"},
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if not task:
        return _redirect("/tasks")
    if status not in TASK_STATUS_ORDER or priority not in TASK_PRIORITIES:
        _set_flash(request, "Некорректный статус или приоритет", "error")
        return _redirect(f"/tasks/{task_id}")

    def parse_date(raw: str):
        return datetime.fromisoformat(raw).date() if raw.strip() else None

    def parse_int(raw: str):
        return int(raw) if raw.strip().isdigit() else None

    try:
        task.title = title.strip()
        task.comment = comment.strip() or None
        task.status = status
        task.priority = priority
        task.tags = tags.strip() or None
        task.start_date = parse_date(start_date)
        task.end_date = parse_date(end_date)
        task.deadline = parse_date(deadline)
        task.assignee_id = parse_int(assignee_id_raw)
        task.queue_id = parse_int(queue_id_raw)
        task.board_id = None
        task.collection_id = parse_int(collection_id_raw)
        task.product_id = parse_int(product_id_raw)
    except ValueError:
        _set_flash(request, "Проверьте даты задачи (формат YYYY-MM-DD)", "error")
        return _redirect(f"/tasks/{task_id}")

    if task.queue_id is None:
        _set_flash(request, "Задача должна принадлежать очереди", "error")
        return _redirect(f"/tasks/{task_id}")

    db.commit()
    _set_flash(request, "Задача обновлена", "success")
    return _redirect(f"/tasks/{task_id}")


@app.post("/tasks/{task_id}/status")
def update_task_status(
    task_id: int,
    request: Request,
    status: str = Form(...),
    _: User = Depends(require_roles("admin", "content-manager")),
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
    _: User = Depends(require_roles("admin", "content-manager")),
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
    user: User = Depends(require_roles("admin", "content-manager")),
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
    _: User = Depends(require_roles("admin", "content-manager")),
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
            "can_manage": user.role in {"admin", "content-manager"},
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
            "can_manage": user.role in {"admin", "content-manager"},
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    db.add(
        Collection(
            code=code.strip(),
            name=name.strip(),
            season=season.strip().upper(),
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    collection = db.get(Collection, collection_id)
    if collection:
        new_is_active = active is not None
        if collection.is_active and not new_is_active:
            usage = _collection_usage_count(db, collection.id)
            if usage:
                _set_flash(request, "Нельзя деактивировать коллекцию: она используется в изделиях", "error")
                return _redirect("/collections")

        collection.code = code.strip()
        collection.name = name.strip()
        collection.season = season.strip().upper()
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
            "can_manage": user.role in {"admin", "content-manager"},
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
            "can_manage": user.role in {"admin", "content-manager"},
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    db.add(
        Supplier(
            code=code.strip(),
            name=name.strip(),
            country=country.strip(),
            contact_email=contact_email.strip() or None,
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    supplier = db.get(Supplier, supplier_id)
    if supplier:
        new_is_active = active is not None
        if supplier.is_active and not new_is_active:
            usage = _supplier_usage_count(db, supplier.id)
            if usage:
                _set_flash(request, "Нельзя деактивировать поставщика: он используется в изделиях", "error")
                return _redirect("/suppliers")

        supplier.code = code.strip()
        supplier.name = name.strip()
        supplier.country = country.strip()
        supplier.contact_email = contact_email.strip() or None
        supplier.is_active = new_is_active
        try:
            db.commit()
            _set_flash(request, "Поставщик обновлен", "success")
        except IntegrityError:
            db.rollback()
            _set_flash(request, "Код поставщика должен быть уникален", "error")
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
    return _render(
        request,
        "dictionaries/detail.html",
        {
            "title": f"Справочник: {dictionary.name}",
            "dictionary": dictionary,
            "items": items,
            "used_by": used_by,
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
    db.add(
        DictionaryItem(
            dictionary_id=dictionary_id,
            code=code.strip(),
            label=label.strip(),
            sort_order=sort_order,
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

    db.add(
        Attribute(
            code=code.strip(),
            name=name.strip(),
            data_type=data_type,
            is_required=is_required is not None,
            is_multivalue=is_multivalue is not None,
            dictionary_id=dictionary_id,
            is_active=True,
        )
    )
    try:
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


@app.get("/products", response_class=HTMLResponse)
def products_page(
    request: Request,
    q: str = "",
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
            "status_filter": status_filter,
            "category_items": category_items,
            "selected_category_id": category_id,
            "collections": collections,
            "selected_collection_id": collection_id,
            "sort": sort,
            "can_manage": user.role in {"admin", "content-manager"},
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
    user: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
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
        sku=sku.strip(),
        name=name.strip(),
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
    active_users = list(db.scalars(select(User).where(User.is_active.is_(True)).order_by(User.login.asc())).all())
    attribute_assignments = list(
        db.scalars(
            select(ProductAttributeAssignment)
            .options(
                joinedload(ProductAttributeAssignment.attribute),
                selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
            )
            .where(ProductAttributeAssignment.product_id == product_id)
            .order_by(ProductAttributeAssignment.id.asc())
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
            "suppliers": suppliers,
            "sample_stages": ["proto", "salesman_sample", "pp_sample", "production"],
            "file_categories": FILE_CATEGORIES,
            "active_users": active_users,
            "attribute_assignments": attribute_assignments,
            "get_value_view": get_value_view,
            "can_manage": user.role in {"admin", "content-manager"},
            "user": user,
        },
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
    user: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")

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

    product.sku = sku.strip()
    product.name = name.strip()
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
    user: User = Depends(require_roles("admin", "content-manager")),
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
    user: User = Depends(require_roles("admin", "content-manager")),
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
    user: User = Depends(require_roles("admin", "content-manager")),
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
def update_product_spec(
    product_id: int,
    request: Request,
    collection_id_raw: str = Form(""),
    supplier_id_raw: str = Form(""),
    style_type: str = Form(""),
    silhouette: str = Form(""),
    fit_type: str = Form(""),
    length_cm_raw: str = Form(""),
    shell_material: str = Form(""),
    lining_material: str = Form(""),
    insulation: str = Form(""),
    sample_stage: str = Form(""),
    planned_cost_raw: str = Form(""),
    actual_cost_raw: str = Form(""),
    user: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")

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

    spec = db.scalar(select(ProductSpec).where(ProductSpec.product_id == product_id))
    if not spec:
        spec = ProductSpec(product_id=product_id)
        db.add(spec)

    spec.collection_id = collection_id
    spec.supplier_id = supplier_id
    spec.style_type = style_type.strip() or None
    spec.silhouette = silhouette.strip() or None
    spec.fit_type = fit_type.strip() or None
    spec.length_cm = length_cm
    spec.shell_material = shell_material.strip() or None
    spec.lining_material = lining_material.strip() or None
    spec.insulation = insulation.strip() or None
    spec.sample_stage = sample_stage.strip() or None
    spec.planned_cost = planned_cost
    spec.actual_cost = actual_cost

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
    user: User = Depends(require_roles("admin", "content-manager")),
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
    user: User = Depends(require_roles("admin", "content-manager")),
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
    user: User = Depends(require_roles("admin", "content-manager")),
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

    assignments = list(
        db.scalars(
            select(ProductAttributeAssignment)
            .options(
                joinedload(ProductAttributeAssignment.attribute),
                selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
            )
            .where(ProductAttributeAssignment.product_id == product_id)
            .order_by(ProductAttributeAssignment.id.asc())
        ).all()
    )
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
            "can_manage": user.role in {"admin", "content-manager"},
            "user": user,
        },
    )


@app.post("/products/{product_id}/attributes/add")
def add_product_attribute(
    product_id: int,
    request: Request,
    attribute_id: int = Form(...),
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    exists = db.execute(
        select(ProductAttributeAssignment.id).where(
            ProductAttributeAssignment.product_id == product_id,
            ProductAttributeAssignment.attribute_id == attribute_id,
        )
    ).first()
    if exists:
        _set_flash(request, "Атрибут уже назначен изделию", "error")
        return _redirect(f"/products/{product_id}/attributes")

    db.add(ProductAttributeAssignment(product_id=product_id, attribute_id=attribute_id))
    db.commit()
    _set_flash(request, "Атрибут назначен изделию", "success")
    return _redirect(f"/products/{product_id}/attributes")


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
            "can_manage": user.role in {"admin", "content-manager"},
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
    user: User = Depends(require_roles("admin", "content-manager")),
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
    _: User = Depends(require_roles("admin", "content-manager")),
    db: Session = Depends(get_db),
):
    assignment = db.scalar(
        select(ProductAttributeAssignment).where(
            ProductAttributeAssignment.id == assignment_id,
            ProductAttributeAssignment.product_id == product_id,
        )
    )
    if assignment:
        db.delete(assignment)
        db.commit()
        _set_flash(request, "Атрибут снят с изделия", "success")
    return _redirect(f"/products/{product_id}/attributes")
