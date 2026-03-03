from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
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
    AuditLog,
    Attribute,
    Collection,
    Dictionary,
    DictionaryItem,
    EntityComment,
    EntityCommentRevision,
    Product,
    ProductAttributeAssignment,
    ProductAttributeValue,
    ProductBOMItem,
    ProductCostingItem,
    ProductFile,
    ProductMaterial,
    ProductSpec,
    ProductSample,
    ProductVariant,
    Supplier,
    SystemSetting,
    Task,
    TaskBoard,
    TaskFile,
    TaskQueue,
    User,
    UserNotification,
    UserNotificationPreference,
)
from .security import hash_password, verify_password
from .services import (
    ATTRIBUTE_GROUPS,
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("app/static/favicon.ico")


@app.middleware("http")
async def force_utf8_response(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html") and "charset" not in content_type.lower():
        response.headers["content-type"] = "text/html; charset=utf-8"
    return response
templates = Jinja2Templates(directory="app/templates")
UPLOAD_DIR = Path("app/static/uploads")
PRODUCT_FILES_DIR = Path("app/static/product_files")
TASK_FILES_DIR = Path("app/static/task_files")
FILE_CATEGORIES = ["sketch", "technical_spec", "patterns", "sample_photo", "materials"]
ATTRIBUTE_GROUP_ORDER = ["service", "fashion_spec", "commercial", "production"]
ATTRIBUTE_GROUP_LABELS = {
    "service": "Служебные",
    "fashion_spec": "Fashion-спецификация",
    "commercial": "Коммерческие",
    "production": "Производственные",
}
ATTRIBUTE_CATALOG_GROUPS = {
    "brand": "service",
    "main_color": "fashion_spec",
    "is_waterproof": "fashion_spec",
    "drop_date": "commercial",
    "retail_price": "commercial",
    "available_sizes": "production",
}
DEPRECATED_ATTRIBUTE_CODES = {"model_name", "style", "capsule", "insulation_level"}
ATTRIBUTE_RESERVED_FIELDS = {
    "sku": "SKU изделия",
    "name": "наименование изделия",
    "modelname": "наименование изделия",
    "названиемодели": "наименование изделия",
    "collection": "поле fashion-спецификации 'Коллекция'",
    "коллекция": "поле fashion-спецификации 'Коллекция'",
    "supplier": "поле fashion-спецификации 'Поставщик'",
    "поставщик": "поле fashion-спецификации 'Поставщик'",
    "style": "поле fashion-спецификации 'Тип изделия'",
    "styletype": "поле fashion-спецификации 'Тип изделия'",
    "типизделия": "поле fashion-спецификации 'Тип изделия'",
    "capsule": "поле fashion-спецификации 'Капсула'",
    "капсула": "поле fashion-спецификации 'Капсула'",
    "silhouette": "поле fashion-спецификации 'Силуэт'",
    "силуэт": "поле fashion-спецификации 'Силуэт'",
    "fittype": "поле fashion-спецификации 'Посадка'",
    "посадка": "поле fashion-спецификации 'Посадка'",
    "lengthcm": "поле fashion-спецификации 'Длина'",
    "длина": "поле fashion-спецификации 'Длина'",
    "shellmaterial": "поле fashion-спецификации 'Основной материал'",
    "основнойматериал": "поле fashion-спецификации 'Основной материал'",
    "liningmaterial": "поле fashion-спецификации 'Подкладка'",
    "подкладка": "поле fashion-спецификации 'Подкладка'",
    "insulation": "поле fashion-спецификации 'Утеплитель'",
    "insulationlevel": "поле fashion-спецификации 'Утеплитель'",
    "утепление": "поле fashion-спецификации 'Утеплитель'",
    "утеплитель": "поле fashion-спецификации 'Утеплитель'",
    "samplestage": "поле fashion-спецификации 'Этап образца'",
    "этапобразца": "поле fashion-спецификации 'Этап образца'",
    "plannedcost": "поле fashion-спецификации 'Плановая себестоимость'",
    "плановаясебестоимость": "поле fashion-спецификации 'Плановая себестоимость'",
    "actualcost": "поле fashion-спецификации 'Фактическая себестоимость'",
    "фактическаясебестоимость": "поле fashion-спецификации 'Фактическая себестоимость'",
}
COMMENT_ENTITY_LABELS = {
    "product": "изделие",
    "collection": "коллекция",
    "supplier": "поставщик",
}
COMMENT_ENTITY_LABELS_RU = {
    "product": "Изделие",
    "collection": "Коллекция",
    "supplier": "Поставщик",
}
NOTIFICATION_EVENT_LABELS = {
    "task_assigned": "Назначение задачи",
    "task_updated": "Обновление задачи",
    "task_status_changed": "Смена статуса задачи",
    "task_file_uploaded": "Файл по задаче",
    "product_updated": "Обновление изделия",
    "product_team_assigned": "Назначение в команду изделия",
    "product_comment": "Комментарий по изделию",
    "collection_comment": "Комментарий по коллекции",
    "supplier_comment": "Комментарий по поставщику",
    "plm_updated": "Изменение PLM-блока",
}
NOTIFICATION_EVENT_DEFAULTS = {
    "task_assigned": True,
    "task_updated": True,
    "task_status_changed": True,
    "task_file_uploaded": True,
    "product_updated": True,
    "product_team_assigned": True,
    "product_comment": True,
    "collection_comment": True,
    "supplier_comment": True,
    "plm_updated": True,
}
DATA_TYPE_LABELS = {
    "string": "Текст",
    "number": "Число",
    "date": "Дата",
    "bool": "Логический",
    "enum": "Справочник",
}
STATUS_LABELS = {
    "draft": "Черновик",
    "active": "Активен",
    "archived": "Архивирован",
}
ROLE_LABELS = {
    "admin": "Администратор",
    "user": "Пользователь",
    "guest": "Гость",
}
FILE_CATEGORY_LABELS = {
    "sketch": "Эскиз",
    "technical_spec": "Тех. спецификация",
    "patterns": "Лекала",
    "sample_photo": "Фото образца",
    "materials": "Материалы",
}
SAMPLE_STAGE_LABELS = {
    "proto": "Пробный образец",
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
    "review": "На проверке",
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
    "plm_block_enabled": "1",
    "plm_bom_enabled": "1",
    "plm_materials_enabled": "1",
    "plm_variants_enabled": "1",
    "plm_samples_enabled": "1",
    "plm_costing_enabled": "1",
    "fashion_block_enabled": "1",
    "fashion_collection_enabled": "1",
    "fashion_supplier_enabled": "1",
    "fashion_style_type_enabled": "1",
    "fashion_capsule_enabled": "1",
    "fashion_sample_stage_enabled": "1",
    "fashion_silhouette_enabled": "1",
    "fashion_fit_type_enabled": "1",
    "fashion_length_cm_enabled": "1",
    "fashion_shell_material_enabled": "1",
    "fashion_lining_material_enabled": "1",
    "fashion_insulation_enabled": "1",
    "fashion_planned_cost_enabled": "1",
    "fashion_actual_cost_enabled": "1",
}
FASHION_FIELD_LABELS = {
    "collection": "Коллекция",
    "supplier": "Поставщик",
    "style_type": "Тип изделия",
    "capsule": "Капсула",
    "sample_stage": "Этап образца",
    "silhouette": "Силуэт",
    "fit_type": "Посадка",
    "length_cm": "Длина (см)",
    "shell_material": "Основной материал",
    "lining_material": "Подкладка",
    "insulation": "Утеплитель",
    "planned_cost": "План. себестоимость",
    "actual_cost": "Факт. себестоимость",
}
SPEC_DICTIONARY_CONFIG: dict[str, dict[str, object]] = {
    "style_type": {
        "code": "product_style_type",
        "name": "\u0422\u0438\u043f \u0438\u0437\u0434\u0435\u043b\u0438\u044f",
        "aliases": ["style_type"],
        "items": [
            ("coat", "\u041f\u0430\u043b\u044c\u0442\u043e"),
            ("trench", "\u0422\u0440\u0435\u043d\u0447"),
            ("puffer", "\u041f\u0443\u0445\u043e\u0432\u0438\u043a"),
            ("jacket", "\u041a\u0443\u0440\u0442\u043a\u0430"),
        ],
    },
    "capsule": {
        "code": "product_capsule",
        "name": "\u041a\u0430\u043f\u0441\u0443\u043b\u0430",
        "aliases": [],
        "items": [
            ("core", "Core"),
            ("studio", "Studio"),
            ("weekend", "Weekend"),
            ("limited", "Limited"),
        ],
    },
    "silhouette": {
        "code": "product_silhouette",
        "name": "\u0421\u0438\u043b\u0443\u044d\u0442",
        "aliases": ["silhouette"],
        "items": [
            ("oversize", "\u041e\u0432\u0435\u0440\u0441\u0430\u0439\u0437"),
            ("straight", "\u041f\u0440\u044f\u043c\u043e\u0439"),
            ("tailored", "\u041f\u0440\u0438\u0442\u0430\u043b\u0435\u043d\u043d\u044b\u0439"),
        ],
    },
    "fit_type": {
        "code": "product_fit_type",
        "name": "\u041f\u043e\u0441\u0430\u0434\u043a\u0430",
        "aliases": ["fit_type", "t_type"],
        "items": [
            ("regular", "Regular"),
            ("relaxed", "Relaxed"),
            ("slim", "Slim"),
        ],
    },
    "shell_material": {
        "code": "product_shell_material",
        "name": "\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
        "aliases": [],
        "items": [
            ("wool", "\u0428\u0435\u0440\u0441\u0442\u044c"),
            ("polyamide", "\u041f\u043e\u043b\u0438\u0430\u043c\u0438\u0434"),
            ("membrane", "\u041c\u0435\u043c\u0431\u0440\u0430\u043d\u0430"),
        ],
    },
    "lining_material": {
        "code": "product_lining_material",
        "name": "\u041f\u043e\u0434\u043a\u043b\u0430\u0434\u043a\u0430",
        "aliases": ["lining_material"],
        "items": [
            ("viscose", "\u0412\u0438\u0441\u043a\u043e\u0437\u0430"),
            ("polyester", "\u041f\u043e\u043b\u0438\u044d\u0441\u0442\u0435\u0440"),
            ("cotton", "\u0425\u043b\u043e\u043f\u043e\u043a"),
        ],
    },
    "insulation": {
        "code": "product_insulation",
        "name": "\u0423\u0442\u0435\u043f\u043b\u0438\u0442\u0435\u043b\u044c",
        "aliases": ["insulation"],
        "items": [
            ("down", "\u041f\u0443\u0445"),
            ("synthetic", "\u0421\u0438\u043d\u0442\u0435\u0442\u0438\u043a\u0430"),
            ("none", "\u0411\u0435\u0437 \u0443\u0442\u0435\u043f\u043b\u0438\u0442\u0435\u043b\u044f"),
        ],
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


def _looks_mojibake(value: str) -> bool:
    if not value:
        return False
    if "\u00a0" in value and ("Р" in value or "С" in value or "Ð" in value):
        return True
    for marker in ("Р ", "РЎ", "Рќ", "Р‚", "РЉ", "Ð", "Ñ", "Ã", "Â", "�"):
        if marker in value:
            return True
    return False


def _fix_mojibake(value: str) -> str:
    if not value or not _looks_mojibake(value):
        return value
    fixed = value
    for _ in range(2):
        try:
            fixed = fixed.encode("cp1251").decode("utf-8")
        except Exception:
            break
        if not _looks_mojibake(fixed):
            return fixed
    return fixed


def _deep_fix(value):
    if isinstance(value, str):
        return _fix_mojibake(value)
    if isinstance(value, list):
        return [_deep_fix(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_fix(item) for item in value)
    if isinstance(value, dict):
        return {key: _deep_fix(val) for key, val in value.items()}
    return value


AUDIT_ENTITY_LABELS = {
    "auth": "Авторизация",
    "settings": "Настройки",
    "user": "Пользователь",
    "task_queue": "Очередь задач",
    "task_board": "Доска",
    "task": "Задача",
    "task_file": "Файл задачи",
    "collection": "Коллекция",
    "supplier": "Поставщик",
    "dictionary": "Справочник",
    "dictionary_item": "Элемент справочника",
    "attribute": "Атрибут",
    "product": "Изделие",
    "product_spec": "Fashion-спецификация",
    "product_team": "Команда изделия",
    "product_file": "Файл изделия",
    "product_attribute": "Атрибут изделия",
}
AUDIT_ACTION_LABELS = {
    "create": "Создание",
    "update": "Изменение",
    "delete": "Удаление",
    "archive": "Архивация",
    "restore": "Восстановление",
    "status_change": "Смена статуса",
    "upload": "Загрузка файла",
    "remove": "Удаление файла",
    "login": "Вход",
    "logout": "Выход",
    "login_failed": "Ошибка входа",
}
AUDIT_FIELD_LABELS = {
    "sku": "SKU",
    "name": "Наименование",
    "description": "Описание",
    "status": "Статус",
    "category_item_id": "Категория",
    "login": "Логин",
    "full_name": "ФИО",
    "department": "Подразделение",
    "position": "Должность",
    "department_item_id": "Подразделение",
    "position_item_id": "Должность",
    "role": "Роль",
    "is_active": "Активность",
    "collection_id": "Коллекция",
    "supplier_id": "Поставщик",
    "style_type": "Тип изделия",
    "capsule": "Капсула",
    "silhouette": "Силуэт",
    "fit_type": "Посадка",
    "length_cm": "Длина",
    "shell_material": "Основной материал",
    "lining_material": "Подкладка",
    "insulation": "Утеплитель",
    "sample_stage": "Этап образца",
    "planned_cost": "Плановая себестоимость",
    "actual_cost": "Фактическая себестоимость",
    "designer_id": "Дизайнер",
    "product_manager_id": "Продукт-менеджер",
    "pattern_maker_id": "Конструктор-модельер",
    "technologist_id": "Технолог",
    "department_head_id": "Руководитель отдела",
    "title": "Название",
    "comment": "Комментарий",
    "priority": "Приоритет",
    "tags": "Теги",
    "start_date": "Начало",
    "end_date": "Окончание",
    "deadline": "Дедлайн",
    "assignee_id": "Исполнитель",
    "queue_id": "Очередь",
    "board_id": "Доска",
    "product_id": "Изделие",
    "code": "Код",
    "data_type": "Тип данных",
    "dictionary_id": "Справочник",
    "is_required": "Обязательный",
    "is_multivalue": "Множественный",
    "sort_order": "Порядок",
    "label": "Значение",
    "country": "Страна",
    "contact_email": "Email",
    "season": "Сезон",
    "year": "Год",
    "brand_line": "Линия",
    "filter_queue_id": "Связанная очередь",
    "category": "Категория файла",
    "original_name": "Файл",
    "title_name": "Название файла",
    "cover_image_path": "Обложка",
    "file_path": "Путь файла",
    "mime_type": "Тип файла",
    "uploaded_by": "Загрузил",
}
AUDIT_SCOPES = {
    "product",
    "task",
    "task_queue",
    "task_board",
    "collection",
    "supplier",
    "dictionary",
    "attribute",
    "user",
    "settings",
}


def _serialize_audit_value(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize_audit_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_audit_value(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize_audit_value(v) for v in value]
    return value


def _audit_details_json(changes: dict | None = None, extra: dict | None = None) -> str | None:
    payload: dict[str, object] = {}
    if changes:
        payload["changes"] = _serialize_audit_value(changes)
    if extra:
        payload["extra"] = _serialize_audit_value(extra)
    if not payload:
        return None
    return json.dumps(payload, ensure_ascii=False)


def _parse_audit_details(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _field_change(before, after) -> dict[str, object]:
    return {"before": _serialize_audit_value(before), "after": _serialize_audit_value(after)}


def _diff_dict(before: dict, after: dict) -> dict[str, dict[str, object]]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    changes: dict[str, dict[str, object]] = {}
    for key in keys:
        if before.get(key) != after.get(key):
            changes[key] = _field_change(before.get(key), after.get(key))
    return changes


def _entity_scope(entity_type: str, entity_id: int | None, scope_type: str | None = None, scope_id: int | None = None) -> tuple[str | None, int | None]:
    if scope_type is not None or scope_id is not None:
        return scope_type, scope_id
    if entity_type in AUDIT_SCOPES:
        return entity_type, entity_id
    return None, None


def _build_audit_summary(action: str, entity_type: str, entity_label: str | None, changes: dict | None = None) -> str:
    action_label = AUDIT_ACTION_LABELS.get(action, action)
    entity_label_text = AUDIT_ENTITY_LABELS.get(entity_type, entity_type)
    subject = entity_label or entity_label_text
    summary = f"{action_label}: {subject}"
    if changes:
        changed_keys = list(changes.keys())
        if changed_keys:
            summary += f" ({', '.join(changed_keys[:4])}"
            if len(changed_keys) > 4:
                summary += ", ..."
            summary += ")"
    return summary


def _write_audit_log(
    db: Session,
    *,
    action: str,
    entity_type: str,
    entity_id: int | None = None,
    entity_label: str | None = None,
    actor: User | None = None,
    actor_login: str | None = None,
    changes: dict | None = None,
    extra: dict | None = None,
    summary: str | None = None,
    scope_type: str | None = None,
    scope_id: int | None = None,
) -> None:
    resolved_scope_type, resolved_scope_id = _entity_scope(entity_type, entity_id, scope_type, scope_id)
    db.add(
        AuditLog(
            entity_type=entity_type,
            entity_id=entity_id,
            entity_label=(entity_label or "").strip() or None,
            scope_type=resolved_scope_type,
            scope_id=resolved_scope_id,
            action=action,
            summary=summary or _build_audit_summary(action, entity_type, entity_label, changes),
            details=_audit_details_json(changes, extra),
            actor_id=actor.id if actor else None,
            actor_login=actor_login or (actor.login if actor else None),
        )
    )


def _audit_rows_with_details(rows: list[AuditLog], db: Session) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for row in rows:
        details = _parse_audit_details(row.details)
        change_items: list[dict[str, str]] = []
        raw_changes = details.get("changes", {})
        if isinstance(raw_changes, dict):
            for field_name, change in raw_changes.items():
                if not isinstance(change, dict):
                    continue
                change_items.append(
                    {
                        "field": field_name,
                        "label": _audit_field_label(field_name),
                        "before": _audit_value_view(db, field_name, change.get("before")),
                        "after": _audit_value_view(db, field_name, change.get("after")),
                    }
                )
        items.append(
            {
                "row": row,
                "details": details,
                "changes": raw_changes,
                "changes_view": change_items,
                "extra": details.get("extra", {}),
            }
        )
    return items


def _user_display(user: User | None) -> str:
    if not user:
        return "-"
    parts = [user.full_name or user.login]
    if user.position:
        parts.append(f"({user.position})")
    return " ".join(parts)


def _product_label(product: Product) -> str:
    return f"{product.sku} - {product.name}"


def _task_label(task: Task) -> str:
    return f"#{task.id} {task.title}"


def _collection_label(collection: Collection) -> str:
    return f"{collection.code} - {collection.name}"


def _supplier_label(supplier: Supplier) -> str:
    return f"{supplier.code} - {supplier.name}"


def _entity_detail_url(entity_type: str, entity_id: int) -> str:
    mapping = {
        "product": f"/products/{entity_id}",
        "collection": f"/collections/{entity_id}",
        "supplier": f"/suppliers/{entity_id}",
        "task": f"/tasks/{entity_id}",
    }
    return mapping.get(entity_type, "/")


def _entity_label(db: Session, entity_type: str, entity_id: int) -> str:
    if entity_type == "product":
        entity = db.get(Product, entity_id)
        return _product_label(entity) if entity else f"Изделие #{entity_id}"
    if entity_type == "collection":
        entity = db.get(Collection, entity_id)
        return _collection_label(entity) if entity else f"Коллекция #{entity_id}"
    if entity_type == "supplier":
        entity = db.get(Supplier, entity_id)
        return _supplier_label(entity) if entity else f"Поставщик #{entity_id}"
    if entity_type == "task":
        entity = db.get(Task, entity_id)
        return _task_label(entity) if entity else f"Задача #{entity_id}"
    return f"{entity_type} #{entity_id}"


def _entity_exists(db: Session, entity_type: str, entity_id: int) -> bool:
    if entity_type == "product":
        return db.get(Product, entity_id) is not None
    if entity_type == "collection":
        return db.get(Collection, entity_id) is not None
    if entity_type == "supplier":
        return db.get(Supplier, entity_id) is not None
    if entity_type == "task":
        return db.get(Task, entity_id) is not None
    return False


def _product_snapshot(product: Product) -> dict[str, object]:
    return {
        "sku": product.sku,
        "name": product.name,
        "description": product.description,
        "status": product.status,
        "category_item_id": product.category_item_id,
    }


def _product_spec_snapshot(spec: ProductSpec | None) -> dict[str, object]:
    if not spec:
        return {}
    return {
        "collection_id": spec.collection_id,
        "supplier_id": spec.supplier_id,
        "style_type": spec.style_type,
        "capsule": spec.capsule,
        "silhouette": spec.silhouette,
        "fit_type": spec.fit_type,
        "length_cm": float(spec.length_cm) if spec.length_cm is not None else None,
        "shell_material": spec.shell_material,
        "lining_material": spec.lining_material,
        "insulation": spec.insulation,
        "sample_stage": spec.sample_stage,
        "planned_cost": float(spec.planned_cost) if spec.planned_cost is not None else None,
        "actual_cost": float(spec.actual_cost) if spec.actual_cost is not None else None,
    }


def _product_team_snapshot(product: Product) -> dict[str, object]:
    return {
        "designer_id": product.designer_id,
        "product_manager_id": product.product_manager_id,
        "pattern_maker_id": product.pattern_maker_id,
        "technologist_id": product.technologist_id,
        "department_head_id": product.department_head_id,
    }


def _task_snapshot(task: Task) -> dict[str, object]:
    return {
        "title": task.title,
        "comment": task.comment,
        "status": task.status,
        "priority": task.priority,
        "tags": task.tags,
        "start_date": task.start_date.isoformat() if task.start_date else None,
        "end_date": task.end_date.isoformat() if task.end_date else None,
        "deadline": task.deadline.isoformat() if task.deadline else None,
        "assignee_id": task.assignee_id,
        "queue_id": task.queue_id,
        "board_id": task.board_id,
        "collection_id": task.collection_id,
        "product_id": task.product_id,
    }


def _user_snapshot(user: User) -> dict[str, object]:
    return {
        "login": user.login,
        "full_name": user.full_name,
        "department": user.department,
        "position": user.position,
        "department_item_id": user.department_item_id,
        "position_item_id": user.position_item_id,
        "role": user.role,
        "is_active": user.is_active,
    }


def _simple_snapshot(entity, fields: list[str]) -> dict[str, object]:
    return {field: _serialize_audit_value(getattr(entity, field)) for field in fields}


def _audit_create_changes(snapshot: dict[str, object]) -> dict[str, dict[str, object]]:
    return {key: _field_change(None, value) for key, value in snapshot.items() if value is not None}


def _audit_delete_changes(snapshot: dict[str, object]) -> dict[str, dict[str, object]]:
    return {key: _field_change(value, None) for key, value in snapshot.items() if value is not None}


def _assignment_value_snapshot(assignment: ProductAttributeAssignment) -> object:
    attr = assignment.attribute
    if not attr or not assignment.values:
        return None
    values: list[object] = []
    for value in assignment.values:
        if attr.data_type == "enum":
            if value.dictionary_item:
                values.append(value.dictionary_item.label)
            elif value.dictionary_item_id is not None:
                values.append(str(value.dictionary_item_id))
            else:
                values.append("")
        else:
            values.append(get_value_view(value, attr.data_type))
    if attr.is_multivalue:
        return values
    return values[0] if values else None


def _product_attributes_snapshot(assignments: list[ProductAttributeAssignment]) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for assignment in assignments:
        if not assignment.attribute:
            continue
        snapshot[assignment.attribute.name] = _assignment_value_snapshot(assignment)
    return snapshot


def _audit_field_label(field_name: str) -> str:
    return AUDIT_FIELD_LABELS.get(field_name, field_name)


def _audit_value_view(db: Session, field_name: str, value) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "Да" if value else "Нет"
    if isinstance(value, list):
        rendered = [_audit_value_view(db, field_name, item) for item in value]
        return ", ".join(item for item in rendered if item and item != "-") or "-"
    if field_name == "status":
        raw = str(value)
        return STATUS_LABELS.get(raw, TASK_STATUS_LABELS.get(raw, raw))
    if field_name == "priority":
        return TASK_PRIORITY_LABELS.get(str(value), str(value))
    if field_name == "role":
        return ROLE_LABELS.get(str(value), str(value))
    if field_name == "data_type":
        return DATA_TYPE_LABELS.get(str(value), str(value))
    if field_name == "sample_stage":
        return SAMPLE_STAGE_LABELS.get(str(value), str(value))
    if field_name in {"category_item_id", "department_item_id", "position_item_id"}:
        item = db.get(DictionaryItem, int(value))
        return item.label if item else str(value)
    if field_name == "collection_id":
        collection = db.get(Collection, int(value))
        if collection:
            return f"{collection.code} - {collection.name}"
    if field_name == "supplier_id":
        supplier = db.get(Supplier, int(value))
        if supplier:
            return f"{supplier.code} - {supplier.name}"
    if field_name in {"designer_id", "product_manager_id", "pattern_maker_id", "technologist_id", "department_head_id", "assignee_id", "author_id", "updated_by", "created_by", "uploaded_by"}:
        user = db.get(User, int(value))
        return _user_display(user) if user else str(value)
    if field_name in {"queue_id", "filter_queue_id"}:
        queue = db.get(TaskQueue, int(value))
        return f"{queue.code} - {queue.name}" if queue else str(value)
    if field_name == "board_id":
        board = db.get(TaskBoard, int(value))
        return f"{board.code} - {board.name}" if board else str(value)
    if field_name == "product_id":
        product = db.get(Product, int(value))
        return _product_label(product) if product else str(value)
    if field_name == "dictionary_id":
        dictionary = db.get(Dictionary, int(value))
        return dictionary.name if dictionary else str(value)
    return str(value)


def _scope_audit_rows(db: Session, scope_type: str, scope_id: int, limit: int = 25) -> list[dict[str, object]]:
    rows = list(
        db.scalars(
            select(AuditLog)
            .options(joinedload(AuditLog.actor))
            .where(AuditLog.scope_type == scope_type, AuditLog.scope_id == scope_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        ).all()
    )
    return _audit_rows_with_details(rows, db)


def _repair_mojibake_db(db: Session) -> int:
    fixed_count = 0

    def fix_obj(obj, fields):
        nonlocal fixed_count
        for field in fields:
            value = getattr(obj, field, None)
            if isinstance(value, str):
                fixed = _fix_mojibake(value)
                if fixed != value:
                    setattr(obj, field, fixed)
                    fixed_count += 1

    for obj in db.scalars(select(Dictionary)).all():
        fix_obj(obj, ["name", "description"])
    for obj in db.scalars(select(DictionaryItem)).all():
        fix_obj(obj, ["label"])
    for obj in db.scalars(select(Attribute)).all():
        fix_obj(obj, ["name", "description"])
    for obj in db.scalars(select(Collection)).all():
        fix_obj(obj, ["name", "description", "brand_line"])
    for obj in db.scalars(select(TaskQueue)).all():
        fix_obj(obj, ["name", "description"])
    for obj in db.scalars(select(TaskBoard)).all():
        fix_obj(obj, ["name", "description"])
    for obj in db.scalars(select(Supplier)).all():
        fix_obj(obj, ["name", "country", "city"])
    for obj in db.scalars(select(Product)).all():
        fix_obj(obj, ["name", "description"])
    for obj in db.scalars(select(ProductFile)).all():
        fix_obj(obj, ["title", "original_name"])
    for obj in db.scalars(select(Task)).all():
        fix_obj(obj, ["title", "comment"])
    for obj in db.scalars(select(User)).all():
        fix_obj(obj, ["full_name", "department", "position"])

    if fixed_count:
        db.commit()
    return fixed_count


def _set_flash(request: Request, message: str, level: str = "info") -> None:
    request.session["flash"] = {"message": _fix_mojibake(message), "level": level}


def _get_flash(request: Request) -> dict[str, str] | None:
    return request.session.pop("flash", None)


def _build_breadcrumbs(path: str, title: str | None) -> list[dict[str, str | None]]:
    raw_title = (title or "").strip() or "\u041a\u0430\u0440\u0442\u043e\u0447\u043a\u0430"
    current_title = _fix_mojibake(raw_title)
    items: list[dict[str, str | None]] = []

    if path == "/login":
        return [{"label": "\u0412\u0445\u043e\u0434", "url": None}]
    if path == "/cabinet":
        return [{"label": "\u041a\u0430\u0431\u0438\u043d\u0435\u0442", "url": None}]
    if path == "/settings":
        return [{"label": "\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438", "url": None}]
    if path == "/audit":
        return [{"label": "\u0410\u0443\u0434\u0438\u0442", "url": None}]

    if path == "/products":
        return [{"label": "\u0418\u0437\u0434\u0435\u043b\u0438\u044f", "url": None}]
    if re.match(r"^/products/\d+$", path):
        return [
            {"label": "\u0418\u0437\u0434\u0435\u043b\u0438\u044f", "url": "/products"},
            {"label": current_title, "url": None},
        ]
    if re.match(r"^/tasks/by-product/\d+$", path):
        return [
            {"label": "\u0418\u0437\u0434\u0435\u043b\u0438\u044f", "url": "/products"},
            {"label": current_title, "url": None},
        ]
    m = re.match(r"^/products/(\d+)/attributes$", path)
    if m:
        return [
            {"label": "\u0418\u0437\u0434\u0435\u043b\u0438\u044f", "url": "/products"},
            {"label": "\u0410\u0442\u0440\u0438\u0431\u0443\u0442\u044b \u0438\u0437\u0434\u0435\u043b\u0438\u044f", "url": None},
        ]
    m = re.match(r"^/products/(\d+)/attributes/\d+$", path)
    if m:
        return [
            {"label": "\u0418\u0437\u0434\u0435\u043b\u0438\u044f", "url": "/products"},
            {"label": "\u0410\u0442\u0440\u0438\u0431\u0443\u0442\u044b \u0438\u0437\u0434\u0435\u043b\u0438\u044f", "url": None},
        ]

    if path == "/collections":
        return [{"label": "\u041a\u043e\u043b\u043b\u0435\u043a\u0446\u0438\u0438", "url": None}]
    if re.match(r"^/collections/\d+$", path):
        return [
            {"label": "\u041a\u043e\u043b\u043b\u0435\u043a\u0446\u0438\u0438", "url": "/collections"},
            {"label": current_title, "url": None},
        ]

    if path == "/suppliers":
        return [{"label": "\u041f\u043e\u0441\u0442\u0430\u0432\u0449\u0438\u043a\u0438", "url": None}]
    if re.match(r"^/suppliers/\d+$", path):
        return [
            {"label": "\u041f\u043e\u0441\u0442\u0430\u0432\u0449\u0438\u043a\u0438", "url": "/suppliers"},
            {"label": current_title, "url": None},
        ]

    if path == "/attributes":
        return [{"label": "\u0410\u0442\u0440\u0438\u0431\u0443\u0442\u044b", "url": None}]
    if re.match(r"^/attributes/\d+$", path):
        return [
            {"label": "\u0410\u0442\u0440\u0438\u0431\u0443\u0442\u044b", "url": "/attributes"},
            {"label": current_title, "url": None},
        ]

    if path == "/dictionaries":
        return [{"label": "\u0421\u043f\u0440\u0430\u0432\u043e\u0447\u043d\u0438\u043a\u0438", "url": None}]
    if re.match(r"^/dictionaries/\d+$", path):
        return [
            {"label": "\u0421\u043f\u0440\u0430\u0432\u043e\u0447\u043d\u0438\u043a\u0438", "url": "/dictionaries"},
            {"label": current_title, "url": None},
        ]

    if path == "/users":
        return [{"label": "\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u0438", "url": None}]
    if re.match(r"^/users/\d+$", path):
        return [
            {"label": "\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u0438", "url": "/users"},
            {"label": current_title, "url": None},
        ]

    if path == "/task-queues":
        return [{"label": "\u041e\u0447\u0435\u0440\u0435\u0434\u0438 \u0437\u0430\u0434\u0430\u0447", "url": None}]
    if re.match(r"^/task-queues/\d+$", path):
        return [
            {"label": "\u041e\u0447\u0435\u0440\u0435\u0434\u0438 \u0437\u0430\u0434\u0430\u0447", "url": "/task-queues"},
            {"label": current_title, "url": None},
        ]

    if path == "/task-boards":
        return [{"label": "\u041a\u0430\u043d\u0431\u0430\u043d", "url": None}]
    if re.match(r"^/task-boards/\d+$", path):
        return [
            {"label": "\u041a\u0430\u043d\u0431\u0430\u043d", "url": "/task-boards"},
            {"label": current_title, "url": None},
        ]

    if path == "/tasks":
        return [{"label": "\u0417\u0430\u0434\u0430\u0447\u0438", "url": None}]
    if re.match(r"^/tasks/\d+$", path):
        return [
            {"label": "\u0417\u0430\u0434\u0430\u0447\u0438", "url": "/tasks"},
            {"label": current_title, "url": None},
        ]

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
    context.setdefault("audit_entity_labels", AUDIT_ENTITY_LABELS)
    context.setdefault("audit_action_labels", AUDIT_ACTION_LABELS)
    context.setdefault("audit_field_labels", AUDIT_FIELD_LABELS)
    if "user" not in context:
        user_id = request.session.get("user_id")
        if user_id:
            db = SessionLocal()
            try:
                context["user"] = db.get(User, user_id)
                context["unread_notifications_count"] = int(
                    db.scalar(
                        select(func.count(UserNotification.id)).where(
                            UserNotification.user_id == user_id,
                            UserNotification.is_read.is_(False),
                        )
                    )
                    or 0
                )
            finally:
                db.close()
        else:
            context["user"] = None
            context["unread_notifications_count"] = 0
    else:
        current_user = context.get("user")
        if current_user and "unread_notifications_count" not in context:
            db = SessionLocal()
            try:
                context["unread_notifications_count"] = int(
                    db.scalar(
                        select(func.count(UserNotification.id)).where(
                            UserNotification.user_id == current_user.id,
                            UserNotification.is_read.is_(False),
                        )
                    )
                    or 0
                )
            finally:
                db.close()
        else:
            context.setdefault("unread_notifications_count", 0)
    context.setdefault("attribute_group_labels", ATTRIBUTE_GROUP_LABELS)
    context.setdefault("notification_event_labels", NOTIFICATION_EVENT_LABELS)
    context = _deep_fix(context)
    return templates.TemplateResponse(template_name, context)


def _redirect(url: str, status_code: int = 303) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=status_code)


def _get_setting(db: Session, key: str, default: str = "") -> str:
    row = db.scalar(select(SystemSetting).where(SystemSetting.key == key))
    return row.value if row else default


def _get_bool_setting(db: Session, key: str, default: bool = False) -> bool:
    raw = _get_setting(db, key, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_plm_settings(db: Session) -> dict[str, bool]:
    return {
        "block_enabled": _get_bool_setting(db, "plm_block_enabled", True),
        "bom_enabled": _get_bool_setting(db, "plm_bom_enabled", True),
        "materials_enabled": _get_bool_setting(db, "plm_materials_enabled", True),
        "variants_enabled": _get_bool_setting(db, "plm_variants_enabled", True),
        "samples_enabled": _get_bool_setting(db, "plm_samples_enabled", True),
        "costing_enabled": _get_bool_setting(db, "plm_costing_enabled", True),
    }


def _get_fashion_settings(db: Session) -> dict[str, bool]:
    return {
        "block_enabled": _get_bool_setting(db, "fashion_block_enabled", True),
        "collection_enabled": _get_bool_setting(db, "fashion_collection_enabled", True),
        "supplier_enabled": _get_bool_setting(db, "fashion_supplier_enabled", True),
        "style_type_enabled": _get_bool_setting(db, "fashion_style_type_enabled", True),
        "capsule_enabled": _get_bool_setting(db, "fashion_capsule_enabled", True),
        "sample_stage_enabled": _get_bool_setting(db, "fashion_sample_stage_enabled", True),
        "silhouette_enabled": _get_bool_setting(db, "fashion_silhouette_enabled", True),
        "fit_type_enabled": _get_bool_setting(db, "fashion_fit_type_enabled", True),
        "length_cm_enabled": _get_bool_setting(db, "fashion_length_cm_enabled", True),
        "shell_material_enabled": _get_bool_setting(db, "fashion_shell_material_enabled", True),
        "lining_material_enabled": _get_bool_setting(db, "fashion_lining_material_enabled", True),
        "insulation_enabled": _get_bool_setting(db, "fashion_insulation_enabled", True),
        "planned_cost_enabled": _get_bool_setting(db, "fashion_planned_cost_enabled", True),
        "actual_cost_enabled": _get_bool_setting(db, "fashion_actual_cost_enabled", True),
    }


def _get_notification_preferences(db: Session, user_id: int) -> dict[str, bool]:
    rows = list(
        db.scalars(select(UserNotificationPreference).where(UserNotificationPreference.user_id == user_id)).all()
    )
    prefs = {key: enabled for key, enabled in NOTIFICATION_EVENT_DEFAULTS.items()}
    for row in rows:
        prefs[row.event_key] = row.enabled
    return prefs


def _notification_enabled(db: Session, user_id: int, event_key: str) -> bool:
    pref = db.scalar(
        select(UserNotificationPreference).where(
            UserNotificationPreference.user_id == user_id,
            UserNotificationPreference.event_key == event_key,
        )
    )
    if pref:
        return pref.enabled
    return NOTIFICATION_EVENT_DEFAULTS.get(event_key, True)


def _create_notification(
    db: Session,
    *,
    user_id: int,
    event_key: str,
    title: str,
    message: str,
    link_url: str | None = None,
    entity_type: str | None = None,
    entity_id: int | None = None,
) -> None:
    if not _notification_enabled(db, user_id, event_key):
        return
    db.add(
        UserNotification(
            user_id=user_id,
            event_key=event_key,
            title=title,
            message=message,
            link_url=link_url,
            entity_type=entity_type,
            entity_id=entity_id,
        )
    )


def _notify_users(
    db: Session,
    user_ids: list[int] | set[int],
    *,
    event_key: str,
    title: str,
    message: str,
    actor_id: int | None = None,
    link_url: str | None = None,
    entity_type: str | None = None,
    entity_id: int | None = None,
) -> None:
    for user_id in sorted({int(v) for v in user_ids if v}):
        if actor_id is not None and user_id == actor_id:
            continue
        _create_notification(
            db,
            user_id=user_id,
            event_key=event_key,
            title=title,
            message=message,
            link_url=link_url,
            entity_type=entity_type,
            entity_id=entity_id,
        )


def _product_watcher_ids(product: Product | None) -> set[int]:
    if not product:
        return set()
    return {
        user_id
        for user_id in [
            product.created_by,
            product.designer_id,
            product.product_manager_id,
            product.pattern_maker_id,
            product.technologist_id,
            product.department_head_id,
            product.updated_by,
        ]
        if user_id
    }


def _task_watcher_ids(task: Task | None) -> set[int]:
    if not task:
        return set()
    return {user_id for user_id in [task.author_id, task.assignee_id] if user_id}


def _comment_rows(db: Session, entity_type: str, entity_id: int) -> list[EntityComment]:
    return list(
        db.scalars(
            select(EntityComment)
            .options(joinedload(EntityComment.author), selectinload(EntityComment.revisions).joinedload(EntityCommentRevision.actor))
            .where(EntityComment.entity_type == entity_type, EntityComment.entity_id == entity_id)
            .order_by(EntityComment.created_at.desc())
        ).all()
    )


def _comment_watchers(db: Session, entity_type: str, entity_id: int) -> set[int]:
    watchers: set[int] = set()
    if entity_type == "product":
        watchers |= _product_watcher_ids(db.get(Product, entity_id))
    elif entity_type == "collection":
        product_ids = list(
            db.scalars(select(Product.id).join(ProductSpec, ProductSpec.product_id == Product.id).where(ProductSpec.collection_id == entity_id)).all()
        )
        for product_id in product_ids:
            watchers |= _product_watcher_ids(db.get(Product, product_id))
    elif entity_type == "supplier":
        product_ids = list(
            db.scalars(select(Product.id).join(ProductSpec, ProductSpec.product_id == Product.id).where(ProductSpec.supplier_id == entity_id)).all()
        )
        for product_id in product_ids:
            watchers |= _product_watcher_ids(db.get(Product, product_id))
    watchers |= {
        user_id
        for user_id in db.scalars(
            select(EntityComment.author_id).where(
                EntityComment.entity_type == entity_type,
                EntityComment.entity_id == entity_id,
                EntityComment.author_id.is_not(None),
            )
        ).all()
        if user_id
    }
    return watchers


def _task_link(task: Task) -> str:
    return f"/tasks/{task.id}"


def _task_reminders(tasks: list[Task]) -> list[dict[str, object]]:
    today = datetime.utcnow().date()
    upcoming_limit = today + timedelta(days=3)
    stale_limit = datetime.utcnow() - timedelta(days=7)
    reminders: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()

    for task in tasks:
        if not task or task.status == "done":
            continue
        if task.deadline:
            if task.deadline < today:
                key = ("overdue", task.id)
                if key not in seen:
                    reminders.append(
                        {
                            "kind": "overdue",
                            "title": f"Просрочена задача #{task.id}",
                            "message": f"{task.title} — дедлайн был {task.deadline.strftime('%Y-%m-%d')}.",
                            "task": task,
                            "link_url": _task_link(task),
                            "sort_key": (0, task.deadline.toordinal(), task.id),
                        }
                    )
                    seen.add(key)
            elif task.deadline <= upcoming_limit:
                key = ("deadline", task.id)
                if key not in seen:
                    reminders.append(
                        {
                            "kind": "deadline",
                            "title": f"Напоминание по дедлайну #{task.id}",
                            "message": f"{task.title} — дедлайн {task.deadline.strftime('%Y-%m-%d')}.",
                            "task": task,
                            "link_url": _task_link(task),
                            "sort_key": (1, task.deadline.toordinal(), task.id),
                        }
                    )
                    seen.add(key)
        if task.updated_at and task.updated_at < stale_limit and task.status in {"backlog", "todo", "in_progress", "review"}:
            key = ("stale", task.id)
            if key not in seen:
                reminders.append(
                    {
                        "kind": "stale",
                        "title": f"Зависшая задача #{task.id}",
                        "message": f"{task.title} — без обновлений с {task.updated_at.strftime('%Y-%m-%d')}.",
                        "task": task,
                        "link_url": _task_link(task),
                        "sort_key": (2, task.updated_at.date().toordinal(), task.id),
                    }
                )
                seen.add(key)

    reminders.sort(key=lambda item: item["sort_key"])
    for item in reminders:
        item.pop("sort_key", None)
    return reminders


def _attribute_groups(assignments: list[ProductAttributeAssignment]) -> list[dict[str, object]]:
    grouped: dict[str, list[ProductAttributeAssignment]] = {key: [] for key in ATTRIBUTE_GROUP_LABELS}
    for assignment in assignments:
        group_code = getattr(assignment.attribute, "group_code", "fashion_spec") if assignment.attribute else "fashion_spec"
        grouped.setdefault(group_code or "fashion_spec", []).append(assignment)
    items: list[dict[str, object]] = []
    for key in ATTRIBUTE_GROUP_ORDER:
        values = grouped.get(key, [])
        if values:
            items.append({"code": key, "label": ATTRIBUTE_GROUP_LABELS.get(key, key), "assignments": values})
    return items


def _catalog_attribute_groups(attributes: list[Attribute]) -> list[dict[str, object]]:
    grouped: dict[str, list[Attribute]] = {key: [] for key in ATTRIBUTE_GROUP_ORDER}
    for attribute in attributes:
        group_code = (attribute.group_code or "fashion_spec").strip() or "fashion_spec"
        grouped.setdefault(group_code, []).append(attribute)
    items: list[dict[str, object]] = []
    for key in ATTRIBUTE_GROUP_ORDER:
        values = grouped.get(key, [])
        if values:
            items.append({"code": key, "label": ATTRIBUTE_GROUP_LABELS.get(key, key), "attributes": values})
    return items


def _normalize_attr_key(value: str) -> str:
    return re.sub(r"[\W_]+", "", (value or "").casefold())


def _attribute_form_error(
    db: Session,
    *,
    code: str,
    name: str,
    data_type: str,
    group_code: str,
    is_multivalue: bool,
    dictionary_id: int | None,
    attribute_id: int | None = None,
) -> str | None:
    code_clean = code.strip()
    name_clean = name.strip()
    if not code_clean or not name_clean:
        return "Код и наименование атрибута обязательны"
    if data_type not in DATA_TYPES:
        return "Некорректный тип данных"
    if group_code not in ATTRIBUTE_GROUPS:
        return "Некорректная группа атрибута"
    if data_type == "enum" and dictionary_id is None:
        return "Для атрибута типа 'Справочник' нужно выбрать справочник"
    if data_type != "enum" and dictionary_id is not None:
        return "Справочник можно указывать только для атрибута типа 'Справочник'"
    if data_type == "bool" and is_multivalue:
        return "Логический атрибут не может быть множественным"

    for raw in {code_clean, name_clean}:
        reserved_label = ATTRIBUTE_RESERVED_FIELDS.get(_normalize_attr_key(raw))
        if reserved_label:
            return f"Атрибут дублирует встроенное {reserved_label}"

    duplicate_stmt = select(Attribute).where(
        or_(
            func.lower(Attribute.code) == code_clean.lower(),
            func.lower(Attribute.name) == name_clean.lower(),
        )
    )
    if attribute_id is not None:
        duplicate_stmt = duplicate_stmt.where(Attribute.id != attribute_id)
    duplicate = db.scalar(duplicate_stmt.limit(1))
    if duplicate:
        if duplicate.code.lower() == code_clean.lower():
            return "Код атрибута должен быть уникальным"
        return "Атрибут с таким наименованием уже существует"
    return None


def _attribute_has_values(db: Session, attribute_id: int) -> bool:
    return not can_change_attribute_type(db, attribute_id)


def _cleanup_empty_attribute_assignments(db: Session) -> int:
    assignments = list(
        db.scalars(
            select(ProductAttributeAssignment)
            .options(joinedload(ProductAttributeAssignment.attribute), selectinload(ProductAttributeAssignment.values))
        ).all()
    )
    removed = 0
    for assignment in assignments:
        if assignment.values:
            continue
        if assignment.attribute and assignment.attribute.is_required:
            continue
        db.delete(assignment)
        removed += 1
    return removed


def _cleanup_invalid_enum_attribute_values(db: Session) -> int:
    values = list(
        db.scalars(
            select(ProductAttributeValue)
            .options(
                joinedload(ProductAttributeValue.dictionary_item),
                joinedload(ProductAttributeValue.assignment).joinedload(ProductAttributeAssignment.attribute),
            )
            .where(ProductAttributeValue.dictionary_item_id.is_not(None))
        ).all()
    )
    removed = 0
    for value in values:
        assignment = value.assignment
        attribute = assignment.attribute if assignment else None
        dictionary_item = value.dictionary_item
        if not assignment or not attribute or attribute.data_type != "enum":
            db.delete(value)
            removed += 1
            continue
        if not dictionary_item or not attribute.dictionary_id or dictionary_item.dictionary_id != attribute.dictionary_id:
            db.delete(value)
            removed += 1
    return removed


def _migrate_redundant_attribute_values(db: Session) -> None:
    target_fields = {
        "style": "style_type",
        "capsule": "capsule",
        "insulation_level": "insulation",
    }
    assignments = list(
        db.scalars(
            select(ProductAttributeAssignment)
            .options(
                joinedload(ProductAttributeAssignment.attribute),
                selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
            )
            .join(ProductAttributeAssignment.attribute)
            .where(Attribute.code.in_(tuple(target_fields)))
        ).all()
    )
    for assignment in assignments:
        attribute = assignment.attribute
        if not attribute:
            continue
        raw_value = _assignment_value_snapshot(assignment)
        if raw_value in (None, "", []):
            continue
        if isinstance(raw_value, list):
            value_text = ", ".join(str(item).strip() for item in raw_value if str(item).strip())
        else:
            value_text = str(raw_value).strip()
        if not value_text:
            continue
        spec = db.scalar(select(ProductSpec).where(ProductSpec.product_id == assignment.product_id))
        if not spec:
            spec = ProductSpec(product_id=assignment.product_id)
            db.add(spec)
            db.flush()
        target_field = target_fields[attribute.code]
        current_value = getattr(spec, target_field)
        if current_value not in (None, ""):
            continue
        setattr(spec, target_field, value_text)


def _remove_deprecated_attributes(db: Session) -> int:
    removed = 0
    deprecated_attributes = list(
        db.scalars(select(Attribute).where(Attribute.code.in_(tuple(DEPRECATED_ATTRIBUTE_CODES)))).all()
    )
    for attribute in deprecated_attributes:
        assignments = list(
            db.scalars(
                select(ProductAttributeAssignment)
                .options(selectinload(ProductAttributeAssignment.values))
                .where(ProductAttributeAssignment.attribute_id == attribute.id)
            ).all()
        )
        for assignment in assignments:
            db.delete(assignment)
        db.flush()
        db.delete(attribute)
        removed += 1
    return removed


def _normalize_attribute_catalog(db: Session) -> None:
    for attribute in db.scalars(select(Attribute)).all():
        target_group = ATTRIBUTE_CATALOG_GROUPS.get(attribute.code)
        if target_group and attribute.group_code != target_group:
            attribute.group_code = target_group
    _cleanup_invalid_enum_attribute_values(db)
    _migrate_redundant_attribute_values(db)
    _remove_deprecated_attributes(db)
    _cleanup_empty_attribute_assignments(db)


def _comment_event_key(entity_type: str) -> str:
    return {
        "product": "product_comment",
        "collection": "collection_comment",
        "supplier": "supplier_comment",
    }.get(entity_type, "product_comment")


def _comment_scope(entity_type: str, entity_id: int) -> tuple[str, int]:
    return entity_type, entity_id


def _save_comment_revision(db: Session, comment: EntityComment, action: str, actor: User, body: str | None = None) -> None:
    db.add(
        EntityCommentRevision(
            comment_id=comment.id,
            body=body if body is not None else comment.body,
            action=action,
            actor_id=actor.id,
        )
    )


def _product_plm_snapshot(db: Session, product_id: int) -> dict[str, object]:
    materials = list(
        db.scalars(select(ProductMaterial).where(ProductMaterial.product_id == product_id).order_by(ProductMaterial.sort_order.asc(), ProductMaterial.id.asc())).all()
    )
    bom_items = list(
        db.scalars(select(ProductBOMItem).where(ProductBOMItem.product_id == product_id).order_by(ProductBOMItem.sort_order.asc(), ProductBOMItem.id.asc())).all()
    )
    variants = list(
        db.scalars(select(ProductVariant).where(ProductVariant.product_id == product_id).order_by(ProductVariant.color.asc(), ProductVariant.size.asc())).all()
    )
    samples = list(
        db.scalars(select(ProductSample).where(ProductSample.product_id == product_id).order_by(ProductSample.id.asc())).all()
    )
    costing_items = list(
        db.scalars(select(ProductCostingItem).where(ProductCostingItem.product_id == product_id).order_by(ProductCostingItem.sort_order.asc(), ProductCostingItem.id.asc())).all()
    )
    return {
        "materials_count": len(materials),
        "bom_count": len(bom_items),
        "variants_count": len(variants),
        "samples_count": len(samples),
        "costing_count": len(costing_items),
        "costing_total": float(sum([item.amount or 0 for item in costing_items] or [0])),
    }


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


def _ensure_attribute_columns() -> None:
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(attributes)").fetchall()
        columns = {row[1] for row in rows}
        if "group_code" not in columns:
            conn.exec_driver_sql("ALTER TABLE attributes ADD COLUMN group_code VARCHAR(40) DEFAULT 'fashion_spec'")
            conn.exec_driver_sql("UPDATE attributes SET group_code = 'fashion_spec' WHERE group_code IS NULL OR trim(group_code) = ''")


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
            canonical.description = f"\u0421\u043f\u0440\u0430\u0432\u043e\u0447\u043d\u0438\u043a \u0434\u043b\u044f \u043f\u043e\u043b\u044f '{canonical_name}'"
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


def _load_product_attribute_assignments(db: Session, product_id: int) -> list[ProductAttributeAssignment]:
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
            .order_by(Attribute.group_code.asc(), Attribute.name.asc(), ProductAttributeAssignment.id.asc())
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

    def cover_for_pdf() -> RLImage | Paragraph:
        if not product.cover_image_path or not product.cover_image_path.startswith("/static/"):
            return p("\u041e\u0431\u043b\u043e\u0436\u043a\u0430 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430")
        local_path = Path("app") / product.cover_image_path.lstrip("/")
        if not local_path.exists() or not local_path.is_file():
            return p("\u041e\u0431\u043b\u043e\u0436\u043a\u0430 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430")
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
                return p("\u041e\u0431\u043b\u043e\u0436\u043a\u0430 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430")
            scale = min((max_w_mm * mm) / src_w, (max_h_mm * mm) / src_h)
            draw_w = max(1, src_w * scale)
            draw_h = max(1, src_h * scale)
            cover = RLImage(buf, width=draw_w, height=draw_h)
            cover.hAlign = "RIGHT"
            return cover
        except Exception:
            return p("\u041e\u0431\u043b\u043e\u0436\u043a\u0430 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430")

    def qr_for_pdf() -> RLImage | Paragraph:
        try:
            url = f"{_get_server_base_url()}/products/{product.id}"
            png = _qr_png(url)
            buf = BytesIO(png)
            qr_img = RLImage(buf, width=35 * mm, height=35 * mm)
            qr_img.hAlign = "RIGHT"
            return qr_img
        except Exception:
            return p("QR \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d")

    flow: list = []
    flow.append(
        Paragraph(
            f"\u041a\u0430\u0440\u0442\u043e\u0447\u043a\u0430 \u0438\u0437\u0434\u0435\u043b\u0438\u044f: {escape(product.sku)}",
            title_style,
        )
    )
    flow.append(Spacer(1, 4 * mm))

    category_label = product.category_item.label if product.category_item else "-"
    details_table = Table(
        [
            ["SKU", product.sku or ""],
            ["\u041d\u0430\u0438\u043c\u0435\u043d\u043e\u0432\u0430\u043d\u0438\u0435", product.name or ""],
            ["\u0421\u0442\u0430\u0442\u0443\u0441", STATUS_LABELS.get(product.status, product.status)],
            ["\u041a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f", category_label],
            ["\u041e\u043f\u0438\u0441\u0430\u043d\u0438\u0435", product.description or "-"],
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

    right_col = Table(
        [[cover_for_pdf()], [Spacer(1, 2 * mm)], [qr_for_pdf()]],
        colWidths=[93 * mm],
    )
    right_col.setStyle(
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

    top_block = Table([[details_table, right_col]], colWidths=[93 * mm, 93 * mm])
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
    flow.append(Paragraph("Fashion-\u0441\u043f\u0435\u0446\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u044f", heading_style))
    spec = product.spec
    spec_rows = [
        [
            "\u041a\u043e\u043b\u043b\u0435\u043a\u0446\u0438\u044f",
            f"{spec.collection.code} - {spec.collection.name}" if spec and spec.collection else "-",
        ],
        ["\u041f\u043e\u0441\u0442\u0430\u0432\u0449\u0438\u043a", spec.supplier.name if spec and spec.supplier else "-"],
        ["\u0422\u0438\u043f \u0438\u0437\u0434\u0435\u043b\u0438\u044f", spec.style_type if spec and spec.style_type else "-"],
        ["\u041a\u0430\u043f\u0441\u0443\u043b\u0430", spec.capsule if spec and spec.capsule else "-"],
        [
            "\u042d\u0442\u0430\u043f \u043e\u0431\u0440\u0430\u0437\u0446\u0430",
            SAMPLE_STAGE_LABELS.get(spec.sample_stage, spec.sample_stage) if spec and spec.sample_stage else "-",
        ],
        ["\u0421\u0438\u043b\u0443\u044d\u0442", spec.silhouette if spec and spec.silhouette else "-"],
        ["\u041f\u043e\u0441\u0430\u0434\u043a\u0430", spec.fit_type if spec and spec.fit_type else "-"],
        [
            "\u0414\u043b\u0438\u043d\u0430 (\u0441\u043c)",
            str(spec.length_cm) if spec and spec.length_cm is not None else "-",
        ],
        [
            "\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
            spec.shell_material if spec and spec.shell_material else "-",
        ],
        ["\u041f\u043e\u0434\u043a\u043b\u0430\u0434\u043a\u0430", spec.lining_material if spec and spec.lining_material else "-"],
        ["\u0423\u0442\u0435\u043f\u043b\u0438\u0442\u0435\u043b\u044c", spec.insulation if spec and spec.insulation else "-"],
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
        flow.append(Paragraph("\u0410\u0442\u0440\u0438\u0431\u0443\u0442\u044b", heading_style))
        attr_rows = [["\u0410\u0442\u0440\u0438\u0431\u0443\u0442", "\u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435"]]
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
    flow.append(Paragraph("\u041a\u043e\u043c\u0430\u043d\u0434\u0430 \u0438\u0437\u0434\u0435\u043b\u0438\u044f", heading_style))
    team_rows = [
        ["\u0414\u0438\u0437\u0430\u0439\u043d\u0435\u0440", (product.designer.full_name or product.designer.login) if product.designer else "-"],
        [
            "\u041f\u0440\u043e\u0434\u0443\u043a\u0442-\u043c\u0435\u043d\u0435\u0434\u0436\u0435\u0440",
            (product.product_manager.full_name or product.product_manager.login) if product.product_manager else "-",
        ],
        [
            "\u041a\u043e\u043d\u0441\u0442\u0440\u0443\u043a\u0442\u043e\u0440-\u043c\u043e\u0434\u0435\u043b\u044c\u0435\u0440",
            (product.pattern_maker.full_name or product.pattern_maker.login) if product.pattern_maker else "-",
        ],
        ["\u0422\u0435\u0445\u043d\u043e\u043b\u043e\u0433", (product.technologist.full_name or product.technologist.login) if product.technologist else "-"],
        [
            "\u0420\u0443\u043a\u043e\u0432\u043e\u0434\u0438\u0442\u0435\u043b\u044c \u043e\u0442\u0434\u0435\u043b\u0430",
            (product.department_head.full_name or product.department_head.login) if product.department_head else "-",
        ],
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
    flow.append(Paragraph("\u0417\u0430\u0434\u0430\u0447\u0438 \u043f\u043e \u0438\u0437\u0434\u0435\u043b\u0438\u044e", heading_style))
    if tasks:
        task_rows = [["ID", "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435", "\u0421\u0442\u0430\u0442\u0443\u0441", "\u041f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442", "\u041e\u0447\u0435\u0440\u0435\u0434\u044c", "\u0414\u0435\u0434\u043b\u0430\u0439\u043d"]]
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
        flow.append(p("\u0417\u0430\u0434\u0430\u0447\u0438 \u043f\u043e \u0438\u0437\u0434\u0435\u043b\u0438\u044e \u043e\u0442\u0441\u0443\u0442\u0441\u0442\u0432\u0443\u044e\u0442."))

    flow.append(Spacer(1, 4 * mm))
    flow.append(Paragraph("\u0424\u0430\u0439\u043b\u044b \u043a\u0430\u0440\u0442\u043e\u0447\u043a\u0438", heading_style))
    if product.files:
        file_rows = [["\u041a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f", "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435", "\u0424\u0430\u0439\u043b"]]
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
        flow.append(p("\u0424\u0430\u0439\u043b\u044b \u043e\u0442\u0441\u0443\u0442\u0441\u0442\u0432\u0443\u044e\u0442."))

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
    _ensure_attribute_columns()
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
            category_dict = Dictionary(code="category", name="Категории", description="Справочник категорий")
            db.add(category_dict)
            db.flush()
            db.add_all(
                [
                    DictionaryItem(dictionary_id=category_dict.id, code="general", label="Общее", sort_order=1),
                    DictionaryItem(dictionary_id=category_dict.id, code="outerwear", label="Верхняя одежда", sort_order=2),
                    DictionaryItem(dictionary_id=category_dict.id, code="accessories", label="Аксессуары", sort_order=3),
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
                ("product", "Продукт"),
                ("pattern", "Конструкторский отдел"),
                ("tech", "Технология"),
                ("management", "Руководство"),
            ]
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
                ("pm", "Продукт-менеджер"),
                ("pattern_maker", "Конструктор-модельер"),
                ("technologist", "Технолог"),
                ("head", "Руководитель отдела"),
            ]
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
            db.add(Collection(code="FW26", name="\u0416\u0435\u043d\u0441\u043a\u0430\u044f \u0432\u0435\u0440\u0445\u043d\u044f\u044f \u043e\u0434\u0435\u0436\u0434\u0430 2026", season="FW", year=2026, brand_line="Women Outerwear"))

        if not db.scalar(select(Supplier).limit(1)):
            db.add(Supplier(code="SUP-001", name="Nord Textile", country="Turkey", contact_email="sales@nord-textile.com"))

        if not db.scalar(select(TaskQueue).limit(1)):
            db.add(TaskQueue(code="FPLM", name="Fashion PLM", description="\u041e\u0447\u0435\u0440\u0435\u0434\u044c \u0437\u0430\u0434\u0430\u0447 Fashion PLM", is_active=True))

        if not db.scalar(select(TaskBoard).limit(1)):
            db.add(TaskBoard(code="DEV", name="\u041a\u043e\u043c\u0430\u043d\u0434\u0430 \u043f\u0440\u043e\u0434\u0443\u043a\u0442\u0430", description="\u0414\u043e\u0441\u043a\u0430 \u0437\u0430\u0434\u0430\u0447 \u043a\u043e\u043c\u0430\u043d\u0434\u044b \u043f\u0440\u043e\u0434\u0443\u043a\u0442\u0430", is_active=True))

        existing_settings = {s.key for s in db.scalars(select(SystemSetting)).all()}
        for key, value in SETTINGS_DEFAULTS.items():
            if key not in existing_settings:
                db.add(SystemSetting(key=key, value=value))

        _repair_mojibake_db(db)
        _normalize_attribute_catalog(db)
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
    return _render(request, "login.html", {"title": "Вход в систему"})


@app.post("/login")
def login(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.scalar(select(User).where(User.login == login))
    if not user or not user.is_active or not verify_password(password, user.password_hash):
        _write_audit_log(
            db,
            action="login_failed",
            entity_type="auth",
            entity_label="Неудачная попытка входа",
            actor_login=login.strip() or None,
            extra={"login": login.strip() or None},
            scope_type="auth",
        )
        db.commit()
        _set_flash(request, "Неверный логин или пароль", "error")
        return _redirect("/login")

    request.session["user_id"] = user.id
    _write_audit_log(
        db,
        action="login",
        entity_type="auth",
        entity_label="Успешный вход",
        actor=user,
        extra={"login": user.login},
        scope_type="auth",
    )
    db.commit()
    _set_flash(request, "Вход выполнен", "success")
    return _redirect("/products")


@app.get("/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")
    if user_id:
        user = db.get(User, user_id)
        if user:
            _write_audit_log(
                db,
                action="logout",
                entity_type="auth",
                entity_label="Выход из системы",
                actor=user,
                extra={"login": user.login},
                scope_type="auth",
            )
            db.commit()
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
    recent_notifications = list(
        db.scalars(
            select(UserNotification)
            .where(UserNotification.user_id == user.id)
            .order_by(UserNotification.created_at.desc())
            .limit(20)
        ).all()
    )
    notification_prefs = _get_notification_preferences(db, user.id)
    task_map: dict[int, Task] = {}
    for task in authored_tasks:
        task_map[task.id] = task
    for task in assigned_tasks:
        task_map[task.id] = task
    task_reminders = _task_reminders(list(task_map.values()))
    return _render(
        request,
        "cabinet.html",
        {
            "title": "Кабинет",
            "authored_tasks": authored_tasks,
            "assigned_tasks": assigned_tasks,
            "products": products,
            "recent_notifications": recent_notifications,
            "notification_prefs": notification_prefs,
            "task_reminders": task_reminders,
            "user": user,
        },
    )


@app.get("/notifications", response_class=HTMLResponse)
def notifications_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    notifications = list(
        db.scalars(
            select(UserNotification)
            .where(UserNotification.user_id == user.id)
            .order_by(UserNotification.created_at.desc())
            .limit(200)
        ).all()
    )
    prefs = _get_notification_preferences(db, user.id)
    return _render(
        request,
        "notifications.html",
        {
            "title": "Уведомления",
            "notifications": notifications,
            "notification_prefs": prefs,
            "user": user,
        },
    )


@app.post("/notifications/preferences/save")
async def save_notification_preferences(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    form = await request.form()
    before = _get_notification_preferences(db, user.id)
    after: dict[str, bool] = {}
    for key in NOTIFICATION_EVENT_DEFAULTS:
        enabled = form.get(f"pref_{key}") is not None
        after[key] = enabled
        row = db.scalar(
            select(UserNotificationPreference).where(
                UserNotificationPreference.user_id == user.id,
                UserNotificationPreference.event_key == key,
            )
        )
        if row:
            row.enabled = enabled
        else:
            db.add(UserNotificationPreference(user_id=user.id, event_key=key, enabled=enabled))
    changes = _diff_dict(before, after)
    if changes:
        _write_audit_log(
            db,
            action="update",
            entity_type="notification",
            entity_label=f"Настройки уведомлений: {user.login}",
            actor=user,
            changes=changes,
            scope_type="user",
            scope_id=user.id,
        )
    db.commit()
    _set_flash(request, "Настройки уведомлений сохранены", "success")
    return _redirect(request.headers.get("referer") or "/notifications")


@app.post("/notifications/{notification_id}/read")
def mark_notification_read(
    notification_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    notification = db.scalar(
        select(UserNotification).where(UserNotification.id == notification_id, UserNotification.user_id == user.id)
    )
    if notification and not notification.is_read:
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        db.commit()
    return _redirect(request.headers.get("referer") or "/notifications")


@app.post("/notifications/read-all")
def mark_all_notifications_read(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    notifications = list(
        db.scalars(
            select(UserNotification).where(
                UserNotification.user_id == user.id,
                UserNotification.is_read.is_(False),
            )
        ).all()
    )
    now = datetime.utcnow()
    for notification in notifications:
        notification.is_read = True
        notification.read_at = now
    if notifications:
        db.commit()
    _set_flash(request, "Уведомления отмечены как прочитанные", "success")
    return _redirect(request.headers.get("referer") or "/notifications")


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
    values["plm_settings"] = _get_plm_settings(db)
    values["fashion_settings"] = _get_fashion_settings(db)
    return _render(
        request,
        "settings.html",
        {
            "title": "Настройки системы",
            "values": values,
            "fashion_field_labels": FASHION_FIELD_LABELS,
            "user": user,
        },
    )


@app.post("/settings")
def update_settings(
    request: Request,
    server_base_url: str = Form(...),
    plm_block_enabled: str | None = Form(None),
    plm_bom_enabled: str | None = Form(None),
    plm_materials_enabled: str | None = Form(None),
    plm_variants_enabled: str | None = Form(None),
    plm_samples_enabled: str | None = Form(None),
    plm_costing_enabled: str | None = Form(None),
    fashion_block_enabled: str | None = Form(None),
    fashion_collection_enabled: str | None = Form(None),
    fashion_supplier_enabled: str | None = Form(None),
    fashion_style_type_enabled: str | None = Form(None),
    fashion_capsule_enabled: str | None = Form(None),
    fashion_sample_stage_enabled: str | None = Form(None),
    fashion_silhouette_enabled: str | None = Form(None),
    fashion_fit_type_enabled: str | None = Form(None),
    fashion_length_cm_enabled: str | None = Form(None),
    fashion_shell_material_enabled: str | None = Form(None),
    fashion_lining_material_enabled: str | None = Form(None),
    fashion_insulation_enabled: str | None = Form(None),
    fashion_planned_cost_enabled: str | None = Form(None),
    fashion_actual_cost_enabled: str | None = Form(None),
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    before = {s.key: s.value for s in db.scalars(select(SystemSetting)).all()}
    incoming = {
        "server_base_url": server_base_url.strip() or SETTINGS_DEFAULTS["server_base_url"],
        "plm_block_enabled": "1" if plm_block_enabled is not None else "0",
        "plm_bom_enabled": "1" if plm_bom_enabled is not None else "0",
        "plm_materials_enabled": "1" if plm_materials_enabled is not None else "0",
        "plm_variants_enabled": "1" if plm_variants_enabled is not None else "0",
        "plm_samples_enabled": "1" if plm_samples_enabled is not None else "0",
        "plm_costing_enabled": "1" if plm_costing_enabled is not None else "0",
        "fashion_block_enabled": "1" if fashion_block_enabled is not None else "0",
        "fashion_collection_enabled": "1" if fashion_collection_enabled is not None else "0",
        "fashion_supplier_enabled": "1" if fashion_supplier_enabled is not None else "0",
        "fashion_style_type_enabled": "1" if fashion_style_type_enabled is not None else "0",
        "fashion_capsule_enabled": "1" if fashion_capsule_enabled is not None else "0",
        "fashion_sample_stage_enabled": "1" if fashion_sample_stage_enabled is not None else "0",
        "fashion_silhouette_enabled": "1" if fashion_silhouette_enabled is not None else "0",
        "fashion_fit_type_enabled": "1" if fashion_fit_type_enabled is not None else "0",
        "fashion_length_cm_enabled": "1" if fashion_length_cm_enabled is not None else "0",
        "fashion_shell_material_enabled": "1" if fashion_shell_material_enabled is not None else "0",
        "fashion_lining_material_enabled": "1" if fashion_lining_material_enabled is not None else "0",
        "fashion_insulation_enabled": "1" if fashion_insulation_enabled is not None else "0",
        "fashion_planned_cost_enabled": "1" if fashion_planned_cost_enabled is not None else "0",
        "fashion_actual_cost_enabled": "1" if fashion_actual_cost_enabled is not None else "0",
    }
    for key, val in incoming.items():
        row = db.scalar(select(SystemSetting).where(SystemSetting.key == key))
        if row:
            row.value = val
        else:
            db.add(SystemSetting(key=key, value=val))
    changes = _diff_dict(before, {**before, **incoming})
    if changes:
        _write_audit_log(
            db,
            action="update",
            entity_type="settings",
            entity_label="Настройки системы",
            actor=user,
            changes=changes,
        )
    db.commit()
    _set_flash(request, "Настройки сохранены", "success")
    return _redirect("/settings")


@app.get("/audit", response_class=HTMLResponse)
def audit_page(
    request: Request,
    entity_type: str = "",
    action: str = "",
    actor_id_raw: str = "",
    scope_type: str = "",
    scope_id_raw: str = "",
    q: str = "",
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    actor_id = int(actor_id_raw) if actor_id_raw.strip().isdigit() else None
    scope_id = int(scope_id_raw) if scope_id_raw.strip().isdigit() else None
    stmt = select(AuditLog).options(joinedload(AuditLog.actor)).order_by(AuditLog.created_at.desc())
    if entity_type:
        stmt = stmt.where(AuditLog.entity_type == entity_type)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if actor_id is not None:
        stmt = stmt.where(AuditLog.actor_id == actor_id)
    if scope_type:
        stmt = stmt.where(AuditLog.scope_type == scope_type)
    if scope_id is not None:
        stmt = stmt.where(AuditLog.scope_id == scope_id)
    if q.strip():
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(
            or_(
                AuditLog.entity_label.ilike(pattern),
                AuditLog.summary.ilike(pattern),
                AuditLog.actor_login.ilike(pattern),
            )
        )

    rows = list(db.scalars(stmt.limit(300)).all())
    actors = list(db.scalars(select(User).order_by(User.login.asc())).all())
    return _render(
        request,
        "audit/list.html",
        {
            "title": "Аудит",
            "audit_rows": _audit_rows_with_details(rows, db),
            "audit_actors": actors,
            "selected_entity_type": entity_type,
            "selected_action": action,
            "selected_actor_id": actor_id,
            "selected_scope_type": scope_type,
            "selected_scope_id": scope_id,
            "q": q,
            "user": user,
        },
    )


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
    user: User = Depends(require_roles("admin")),
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

    new_user = User(
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
    db.add(new_user)
    try:
        db.flush()
        _write_audit_log(
            db,
            action="create",
            entity_type="user",
            entity_id=new_user.id,
            entity_label=new_user.login,
            actor=user,
            changes={key: {"before": None, "after": value} for key, value in _user_snapshot(new_user).items() if value is not None},
        )
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
        before = _user_snapshot(target)
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
        changes = _diff_dict(before, _user_snapshot(target))
        _write_audit_log(
            db,
            action="update",
            entity_type="user",
            entity_id=target.id,
            entity_label=target.login,
            actor=user,
            changes=changes,
            summary=f"Статус пользователя изменен: {target.login}",
        )
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
        before = _user_snapshot(target)
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
        _write_audit_log(
            db,
            action="update",
            entity_type="user",
            entity_id=target.id,
            entity_label=target.login,
            actor=user,
            changes=_diff_dict(before, _user_snapshot(target)),
            summary=f"Роль пользователя изменена: {target.login}",
        )
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
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    redirect_to = (return_to or request.headers.get("referer") or f"/users/{user_id}").strip()
    if not redirect_to.startswith("/"):
        redirect_to = f"/users/{user_id}"

    target = db.get(User, user_id)
    if not target:
        _set_flash(request, "Пользователь не найден", "error")
        return _redirect("/users")
    before = _user_snapshot(target)
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
        changes = _diff_dict(before, _user_snapshot(target))
        if changes:
            _write_audit_log(
                db,
                action="update",
                entity_type="user",
                entity_id=target.id,
                entity_label=target.login,
                actor=user,
                changes=changes,
                summary=f"Профиль пользователя обновлен: {target.login}",
            )
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
            reasons.append(f"Р В РЎвЂР В Р’В·Р В РўвЂР В Р’ВµР В Р’В»Р В РЎвЂР РЋР РЏ (Р В РЎвЂќР В РЎвЂўР В РЎВР В Р’В°Р В Р вЂ¦Р В РўвЂР В Р’В°, {usage['products_team']})")
        if usage["tasks_owner"]:
            reasons.append(f"Р В Р’В·Р В Р’В°Р В РўвЂР В Р’В°Р РЋРІР‚РЋР В РЎвЂ ({usage['tasks_owner']})")
        _set_flash(request, f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р В РЎвЂ”Р В РЎвЂўР В Р’В»Р РЋР Р‰Р В Р’В·Р В РЎвЂўР В Р вЂ Р В Р’В°Р РЋРІР‚С™Р В Р’ВµР В Р’В»Р РЋР РЏ: Р В Р’ВµР РЋР С“Р РЋРІР‚С™Р РЋР Р‰ Р РЋР С“Р В Р вЂ Р РЋР РЏР В Р’В·Р В Р’В°Р В Р вЂ¦Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ Р В РўвЂР В Р’В°Р В Р вЂ¦Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ ({', '.join(reasons)}).", "error")
        return _redirect("/users")

    db.query(Product).filter(Product.created_by == target.id).update({Product.created_by: None}, synchronize_session=False)
    db.query(Product).filter(Product.updated_by == target.id).update({Product.updated_by: None}, synchronize_session=False)
    db.query(ProductFile).filter(ProductFile.uploaded_by == target.id).update(
        {ProductFile.uploaded_by: None}, synchronize_session=False
    )
    db.query(TaskFile).filter(TaskFile.uploaded_by == target.id).update({TaskFile.uploaded_by: None}, synchronize_session=False)

    _write_audit_log(
        db,
        action="delete",
        entity_type="user",
        entity_id=target.id,
        entity_label=target.login,
        actor=user,
        changes={key: {"before": value, "after": None} for key, value in _user_snapshot(target).items() if value is not None},
        summary=f"Пользователь удален: {target.login}",
    )
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
            "title": f"Очередь задач: {queue.name}",
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
    user: User = Depends(require_roles("admin", "user")),
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
        queue = TaskQueue(
            code=next_code,
            name=name_clean,
            description=description.strip() or None,
            is_active=active is not None,
        )
        db.add(queue)
        try:
            db.flush()
            _write_audit_log(
                db,
                action="create",
                entity_type="task_queue",
                entity_id=queue.id,
                entity_label=queue.name,
                actor=user,
                changes={key: {"before": None, "after": value} for key, value in _simple_snapshot(queue, ["code", "name", "description", "is_active"]).items() if value is not None},
            )
            db.commit()
            _set_flash(request, f"Р В РЎвЂєР РЋРІР‚РЋР В Р’ВµР РЋР вЂљР В Р’ВµР В РўвЂР РЋР Р‰ Р РЋР С“Р В РЎвЂўР В Р’В·Р В РўвЂР В Р’В°Р В Р вЂ¦Р В Р’В° ({next_code})", "success")
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
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    name_clean = name.strip()
    if not name_clean:
        _set_flash(request, "Название очереди обязательно", "error")
        return _redirect("/queues")
    queue = db.get(TaskQueue, queue_id)
    if queue:
        before = _simple_snapshot(queue, ["code", "name", "description", "is_active"])
        new_is_active = active is not None
        if queue.is_active and not new_is_active and _queue_usage_count(db, queue.id):
            _set_flash(request, "Нельзя деактивировать очередь: она используется в задачах", "error")
            return _redirect("/queues")

        queue.name = name_clean
        queue.description = description.strip() or None
        queue.is_active = new_is_active
        try:
            changes = _diff_dict(before, _simple_snapshot(queue, ["code", "name", "description", "is_active"]))
            if changes:
                _write_audit_log(
                    db,
                    action="update",
                    entity_type="task_queue",
                    entity_id=queue.id,
                    entity_label=queue.name,
                    actor=user,
                    changes=changes,
                )
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
    user: User = Depends(require_roles("admin", "user")),
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
            reasons.append(f"Р В Р’В·Р В Р’В°Р В РўвЂР В Р’В°Р РЋРІР‚РЋР В РЎвЂ ({task_usage})")
        if board_usage:
            reasons.append(f"Р В РўвЂР В РЎвЂўР РЋР С“Р В РЎвЂќР В РЎвЂ ({board_usage})")
        _set_flash(request, f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р В РЎвЂўР РЋРІР‚РЋР В Р’ВµР РЋР вЂљР В Р’ВµР В РўвЂР РЋР Р‰: Р В Р’ВµР РЋР С“Р РЋРІР‚С™Р РЋР Р‰ Р РЋР С“Р В Р вЂ Р РЋР РЏР В Р’В·Р В Р’В°Р В Р вЂ¦Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ Р В РўвЂР В Р’В°Р В Р вЂ¦Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ ({', '.join(reasons)}).", "error")
        return _redirect(f"/queues/{queue_id}")

    _write_audit_log(
        db,
        action="delete",
        entity_type="task_queue",
        entity_id=queue.id,
        entity_label=queue.name,
        actor=user,
        changes={key: {"before": value, "after": None} for key, value in _simple_snapshot(queue, ["code", "name", "description", "is_active"]).items() if value is not None},
    )
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
            "title": "Канбан-доски",
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
    user: User = Depends(require_roles("admin", "user")),
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
        board = TaskBoard(
            code=next_code,
            name=name_clean,
            description=description.strip() or None,
            filter_queue_id=filter_queue_id,
            is_active=active is not None,
        )
        db.add(board)
        try:
            db.flush()
            _write_audit_log(
                db,
                action="create",
                entity_type="task_board",
                entity_id=board.id,
                entity_label=board.name,
                actor=user,
                changes={key: {"before": None, "after": value} for key, value in _simple_snapshot(board, ["code", "name", "description", "filter_queue_id", "is_active"]).items() if value is not None},
            )
            db.commit()
            _set_flash(request, f"Р В РІР‚СњР В РЎвЂўР РЋР С“Р В РЎвЂќР В Р’В° Р РЋР С“Р В РЎвЂўР В Р’В·Р В РўвЂР В Р’В°Р В Р вЂ¦Р В Р’В° ({next_code})", "success")
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
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    name_clean = name.strip()
    if not name_clean:
        _set_flash(request, "Название доски обязательно", "error")
        return _redirect(f"/boards/{board_id}")
    board = db.scalar(select(TaskBoard).options(joinedload(TaskBoard.filter_queue)).where(TaskBoard.id == board_id))
    if board:
        before = _simple_snapshot(board, ["code", "name", "description", "filter_queue_id", "is_active"])
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
            changes = _diff_dict(before, _simple_snapshot(board, ["code", "name", "description", "filter_queue_id", "is_active"]))
            if changes:
                _write_audit_log(
                    db,
                    action="update",
                    entity_type="task_board",
                    entity_id=board.id,
                    entity_label=board.name,
                    actor=user,
                    changes=changes,
                )
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
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    board = db.get(TaskBoard, board_id)
    if not board:
        return _redirect("/boards")

    usage = _board_usage_count(db, board.id)
    if usage:
        _set_flash(request, f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р В РўвЂР В РЎвЂўР РЋР С“Р В РЎвЂќР РЋРЎвЂњ: Р В Р’ВµР РЋР С“Р РЋРІР‚С™Р РЋР Р‰ Р РЋР С“Р В Р вЂ Р РЋР РЏР В Р’В·Р В Р’В°Р В Р вЂ¦Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ Р В Р’В·Р В Р’В°Р В РўвЂР В Р’В°Р РЋРІР‚РЋР В РЎвЂ ({usage}).", "error")
        return _redirect(f"/boards/{board_id}")

    _write_audit_log(
        db,
        action="delete",
        entity_type="task_board",
        entity_id=board.id,
        entity_label=board.name,
        actor=user,
        changes={key: {"before": value, "after": None} for key, value in _simple_snapshot(board, ["code", "name", "description", "filter_queue_id", "is_active"]).items() if value is not None},
    )
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
    db.flush()
    _write_audit_log(
        db,
        action="create",
        entity_type="task",
        entity_id=task.id,
        entity_label=_task_label(task),
        actor=user,
        changes=_audit_create_changes(_task_snapshot(task)),
        scope_type="product" if task.product_id else None,
        scope_id=task.product_id,
    )
    _notify_users(
        db,
        _task_watcher_ids(task),
        event_key="task_assigned",
        title="Новая задача",
        message=f"{user.full_name or user.login} создал задачу '{task.title}'.",
        actor_id=user.id,
        link_url=f"/tasks/{task.id}",
        entity_type="task",
        entity_id=task.id,
    )
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
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if not task:
        return _redirect("/tasks")
    before = _task_snapshot(task)
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

    changes = _diff_dict(before, _task_snapshot(task))
    if changes:
        scope_product_id = task.product_id or before.get("product_id")
        _write_audit_log(
            db,
            action="update",
            entity_type="task",
            entity_id=task.id,
            entity_label=_task_label(task),
            actor=user,
            changes=changes,
            scope_type="product" if scope_product_id else None,
            scope_id=scope_product_id,
        )
        _notify_users(
            db,
            _task_watcher_ids(task) | ({int(before["assignee_id"])} if before.get("assignee_id") else set()),
            event_key="task_updated",
            title="Задача обновлена",
            message=f"{user.full_name or user.login} обновил задачу '{task.title}'.",
            actor_id=user.id,
            link_url=f"/tasks/{task.id}",
            entity_type="task",
            entity_id=task.id,
        )
    db.commit()
    _set_flash(request, "Задача обновлена", "success")
    return _redirect(f"/tasks/{task_id}")


@app.post("/tasks/{task_id}/status")
def update_task_status(
    task_id: int,
    request: Request,
    status: str = Form(...),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if task and status in TASK_STATUS_ORDER:
        before_status = task.status
        task.status = status
        if before_status != status:
            _write_audit_log(
                db,
                action="status_change",
                entity_type="task",
                entity_id=task.id,
                entity_label=_task_label(task),
                actor=user,
                changes={"status": _field_change(before_status, status)},
                scope_type="product" if task.product_id else None,
                scope_id=task.product_id,
            )
            _notify_users(
                db,
                _task_watcher_ids(task),
                event_key="task_status_changed",
                title="Статус задачи изменен",
                message=f"{user.full_name or user.login} изменил статус задачи '{task.title}'.",
                actor_id=user.id,
                link_url=f"/tasks/{task.id}",
                entity_type="task",
                entity_id=task.id,
            )
        db.commit()
        _set_flash(request, "Статус задачи обновлен", "success")
    return _redirect(request.headers.get("referer") or "/tasks")


@app.post("/tasks/{task_id}/status-json")
def update_task_status_json(
    task_id: int,
    status: str = Form(...),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    task = db.get(Task, task_id)
    if not task:
        return JSONResponse({"ok": False, "error": "task_not_found"}, status_code=404)
    if status not in TASK_STATUS_ORDER:
        return JSONResponse({"ok": False, "error": "invalid_status"}, status_code=400)
    before_status = task.status
    task.status = status
    if before_status != status:
        _write_audit_log(
            db,
            action="status_change",
            entity_type="task",
            entity_id=task.id,
            entity_label=_task_label(task),
            actor=user,
            changes={"status": _field_change(before_status, status)},
            scope_type="product" if task.product_id else None,
            scope_id=task.product_id,
        )
        _notify_users(
            db,
            _task_watcher_ids(task),
            event_key="task_status_changed",
            title="Статус задачи изменен",
            message=f"{user.full_name or user.login} изменил статус задачи '{task.title}'.",
            actor_id=user.id,
            link_url=f"/tasks/{task.id}",
            entity_type="task",
            entity_id=task.id,
        )
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

    record = TaskFile(
        task_id=task_id,
        original_name=task_file.filename or file_name,
        file_path=f"/static/task_files/{file_name}",
        mime_type=task_file.content_type,
        uploaded_by=user.id,
    )
    db.add(record)
    db.flush()
    _write_audit_log(
        db,
        action="upload",
        entity_type="task_file",
        entity_id=record.id,
        entity_label=record.original_name,
        actor=user,
        changes=_audit_create_changes(
            _simple_snapshot(record, ["original_name", "file_path", "mime_type", "uploaded_by"])
        ),
        scope_type="product" if task.product_id else "task",
        scope_id=task.product_id or task.id,
    )
    _notify_users(
        db,
        _task_watcher_ids(task),
        event_key="task_file_uploaded",
        title="Файл по задаче",
        message=f"{user.full_name or user.login} добавил файл к задаче '{task.title}'.",
        actor_id=user.id,
        link_url=f"/tasks/{task.id}",
        entity_type="task",
        entity_id=task.id,
    )
    db.commit()
    _set_flash(request, "Файл прикреплен к задаче", "success")
    return _redirect(f"/tasks/{task_id}")


@app.post("/tasks/{task_id}/files/{file_id}/delete")
def delete_task_file(
    task_id: int,
    file_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    rec = db.scalar(select(TaskFile).where(TaskFile.id == file_id, TaskFile.task_id == task_id))
    if not rec:
        return _redirect(f"/tasks/{task_id}")
    task = db.get(Task, task_id)
    snapshot = _simple_snapshot(rec, ["original_name", "file_path", "mime_type", "uploaded_by"])
    if rec.file_path.startswith("/static/task_files/"):
        file_name = rec.file_path.split("/static/task_files/", 1)[1]
        path = TASK_FILES_DIR / file_name
        if path.exists():
            path.unlink()
    _write_audit_log(
        db,
        action="remove",
        entity_type="task_file",
        entity_id=rec.id,
        entity_label=rec.original_name,
        actor=user,
        changes=_audit_delete_changes(snapshot),
        scope_type="product" if task and task.product_id else "task",
        scope_id=(task.product_id if task and task.product_id else task_id),
    )
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
    comments = _comment_rows(db, "collection", collection_id)
    collection_audit_rows = _scope_audit_rows(db, "collection", collection_id, 20)
    return _render(
        request,
        "collections/detail.html",
        {
            "title": f"Коллекция: {collection.name}",
            "collection": collection,
            "products": products,
            "comments": comments,
            "collection_audit_rows": collection_audit_rows,
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
    user: User = Depends(require_roles("admin", "user")),
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
    collection = Collection(
        code=code_clean,
        name=name_clean,
        season=season_clean,
        year=year,
        brand_line=brand_line.strip() or None,
        is_active=active is not None,
    )
    db.add(collection)
    try:
        db.flush()
        _write_audit_log(
            db,
            action="create",
            entity_type="collection",
            entity_id=collection.id,
            entity_label=collection.name,
            actor=user,
            changes=_audit_create_changes(
                _simple_snapshot(collection, ["code", "name", "season", "year", "brand_line", "is_active"])
            ),
        )
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
    user: User = Depends(require_roles("admin", "user")),
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
        before = _simple_snapshot(collection, ["code", "name", "season", "year", "brand_line", "is_active"])
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
            changes = _diff_dict(before, _simple_snapshot(collection, ["code", "name", "season", "year", "brand_line", "is_active"]))
            if changes:
                _write_audit_log(
                    db,
                    action="update",
                    entity_type="collection",
                    entity_id=collection.id,
                    entity_label=collection.name,
                    actor=user,
                    changes=changes,
                )
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
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    collection = db.get(Collection, collection_id)
    if not collection:
        return _redirect("/collections")

    usage = _collection_usage_count(db, collection.id)
    if usage:
        _set_flash(request, f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р В РЎвЂќР В РЎвЂўР В Р’В»Р В Р’В»Р В Р’ВµР В РЎвЂќР РЋРІР‚В Р В РЎвЂР РЋР вЂ№: Р В Р’ВµР РЋР С“Р РЋРІР‚С™Р РЋР Р‰ Р РЋР С“Р В Р вЂ Р РЋР РЏР В Р’В·Р В Р’В°Р В Р вЂ¦Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ Р В РЎвЂР В Р’В·Р В РўвЂР В Р’ВµР В Р’В»Р В РЎвЂР РЋР РЏ ({usage}).", "error")
        return _redirect(f"/collections/{collection_id}")

    _write_audit_log(
        db,
        action="delete",
        entity_type="collection",
        entity_id=collection.id,
        entity_label=collection.name,
        actor=user,
        changes=_audit_delete_changes(
            _simple_snapshot(collection, ["code", "name", "season", "year", "brand_line", "is_active"])
        ),
    )
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
    comments = _comment_rows(db, "supplier", supplier_id)
    supplier_audit_rows = _scope_audit_rows(db, "supplier", supplier_id, 20)
    return _render(
        request,
        "suppliers/detail.html",
        {
            "title": f"Поставщик: {supplier.name}",
            "supplier": supplier,
            "products": products,
            "comments": comments,
            "supplier_audit_rows": supplier_audit_rows,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/comments/{entity_type}/{entity_id}")
def create_entity_comment(
    entity_type: str,
    entity_id: int,
    request: Request,
    body: str = Form(...),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    if entity_type not in COMMENT_ENTITY_LABELS:
        _set_flash(request, "Некорректный тип сущности для комментария", "error")
        return _redirect(request.headers.get("referer") or "/")
    if not _entity_exists(db, entity_type, entity_id):
        _set_flash(request, "Сущность для комментария не найдена", "error")
        return _redirect(request.headers.get("referer") or "/")
    body_clean = body.strip()
    if not body_clean:
        _set_flash(request, "Комментарий не может быть пустым", "error")
        return _redirect(request.headers.get("referer") or _entity_detail_url(entity_type, entity_id))
    comment = EntityComment(entity_type=entity_type, entity_id=entity_id, body=body_clean, author_id=user.id)
    db.add(comment)
    db.flush()
    _save_comment_revision(db, comment, "create", user, body_clean)
    entity_label = _entity_label(db, entity_type, entity_id)
    _write_audit_log(
        db,
        action="create",
        entity_type="comment",
        entity_id=comment.id,
        entity_label=entity_label,
        actor=user,
        changes=_audit_create_changes({"body": body_clean}),
        summary=f"Комментарий добавлен: {entity_label}",
        scope_type=entity_type,
        scope_id=entity_id,
    )
    _notify_users(
        db,
        _comment_watchers(db, entity_type, entity_id),
        event_key=_comment_event_key(entity_type),
        title=f"Новый комментарий: {COMMENT_ENTITY_LABELS_RU.get(entity_type, entity_type)}",
        message=f"{user.full_name or user.login} добавил комментарий по сущности '{entity_label}'.",
        actor_id=user.id,
        link_url=_entity_detail_url(entity_type, entity_id),
        entity_type=entity_type,
        entity_id=entity_id,
    )
    db.commit()
    _set_flash(request, "Комментарий добавлен", "success")
    return _redirect(request.headers.get("referer") or _entity_detail_url(entity_type, entity_id))


@app.post("/comments/{comment_id}/update")
def update_entity_comment(
    comment_id: int,
    request: Request,
    body: str = Form(...),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    comment = db.get(EntityComment, comment_id)
    if not comment or comment.is_deleted:
        _set_flash(request, "Комментарий не найден", "error")
        return _redirect(request.headers.get("referer") or "/")
    if user.role != "admin" and comment.author_id != user.id:
        _set_flash(request, "Редактировать комментарий может только автор", "error")
        return _redirect(request.headers.get("referer") or _entity_detail_url(comment.entity_type, comment.entity_id))
    body_clean = body.strip()
    if not body_clean:
        _set_flash(request, "Комментарий не может быть пустым", "error")
        return _redirect(request.headers.get("referer") or _entity_detail_url(comment.entity_type, comment.entity_id))
    before = comment.body
    comment.body = body_clean
    comment.updated_at = datetime.utcnow()
    _save_comment_revision(db, comment, "update", user, body_clean)
    entity_label = _entity_label(db, comment.entity_type, comment.entity_id)
    _write_audit_log(
        db,
        action="update",
        entity_type="comment",
        entity_id=comment.id,
        entity_label=entity_label,
        actor=user,
        changes={"body": _field_change(before, body_clean)},
        summary=f"Комментарий обновлен: {entity_label}",
        scope_type=comment.entity_type,
        scope_id=comment.entity_id,
    )
    db.commit()
    _set_flash(request, "Комментарий обновлен", "success")
    return _redirect(request.headers.get("referer") or _entity_detail_url(comment.entity_type, comment.entity_id))


@app.post("/comments/{comment_id}/delete")
def delete_entity_comment(
    comment_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    comment = db.get(EntityComment, comment_id)
    if not comment or comment.is_deleted:
        _set_flash(request, "Комментарий не найден", "error")
        return _redirect(request.headers.get("referer") or "/")
    if user.role != "admin" and comment.author_id != user.id:
        _set_flash(request, "Удалять комментарий может только автор", "error")
        return _redirect(request.headers.get("referer") or _entity_detail_url(comment.entity_type, comment.entity_id))
    before = comment.body
    comment.is_deleted = True
    comment.body = "[удалено]"
    comment.updated_at = datetime.utcnow()
    _save_comment_revision(db, comment, "delete", user, before)
    entity_label = _entity_label(db, comment.entity_type, comment.entity_id)
    _write_audit_log(
        db,
        action="delete",
        entity_type="comment",
        entity_id=comment.id,
        entity_label=entity_label,
        actor=user,
        changes={"body": _field_change(before, None)},
        summary=f"Комментарий удален: {entity_label}",
        scope_type=comment.entity_type,
        scope_id=comment.entity_id,
    )
    db.commit()
    _set_flash(request, "Комментарий удален", "success")
    return _redirect(request.headers.get("referer") or _entity_detail_url(comment.entity_type, comment.entity_id))


@app.post("/suppliers")
def create_supplier(
    request: Request,
    code: str = Form(...),
    name: str = Form(...),
    country: str = Form(...),
    contact_email: str = Form(""),
    active: str | None = Form(None),
    user: User = Depends(require_roles("admin", "user")),
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
    supplier = Supplier(
        code=code_clean,
        name=name_clean,
        country=country_clean,
        contact_email=email_clean or None,
        is_active=active is not None,
    )
    db.add(supplier)
    try:
        db.flush()
        _write_audit_log(
            db,
            action="create",
            entity_type="supplier",
            entity_id=supplier.id,
            entity_label=supplier.name,
            actor=user,
            changes=_audit_create_changes(
                _simple_snapshot(supplier, ["code", "name", "country", "contact_email", "is_active"])
            ),
        )
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
    user: User = Depends(require_roles("admin", "user")),
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
        before = _simple_snapshot(supplier, ["code", "name", "country", "contact_email", "is_active"])
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
            changes = _diff_dict(before, _simple_snapshot(supplier, ["code", "name", "country", "contact_email", "is_active"]))
            if changes:
                _write_audit_log(
                    db,
                    action="update",
                    entity_type="supplier",
                    entity_id=supplier.id,
                    entity_label=supplier.name,
                    actor=user,
                    changes=changes,
                )
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
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    supplier = db.get(Supplier, supplier_id)
    if not supplier:
        return _redirect("/suppliers")

    usage = _supplier_usage_count(db, supplier.id)
    if usage:
        _set_flash(request, f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р В РЎвЂ”Р В РЎвЂўР РЋР С“Р РЋРІР‚С™Р В Р’В°Р В Р вЂ Р РЋРІР‚В°Р В РЎвЂР В РЎвЂќР В Р’В°: Р В Р’ВµР РЋР С“Р РЋРІР‚С™Р РЋР Р‰ Р РЋР С“Р В Р вЂ Р РЋР РЏР В Р’В·Р В Р’В°Р В Р вЂ¦Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ Р В РЎвЂР В Р’В·Р В РўвЂР В Р’ВµР В Р’В»Р В РЎвЂР РЋР РЏ ({usage}).", "error")
        return _redirect(f"/suppliers/{supplier_id}")

    _write_audit_log(
        db,
        action="delete",
        entity_type="supplier",
        entity_id=supplier.id,
        entity_label=supplier.name,
        actor=user,
        changes=_audit_delete_changes(
            _simple_snapshot(supplier, ["code", "name", "country", "contact_email", "is_active"])
        ),
    )
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
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    dictionary = Dictionary(code=code.strip(), name=name.strip(), description=description.strip() or None)
    db.add(dictionary)
    try:
        db.flush()
        _write_audit_log(
            db,
            action="create",
            entity_type="dictionary",
            entity_id=dictionary.id,
            entity_label=dictionary.name,
            actor=user,
            changes=_audit_create_changes(_simple_snapshot(dictionary, ["code", "name", "description"])),
        )
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
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    dictionary = db.get(Dictionary, dictionary_id)
    if dictionary:
        before = _simple_snapshot(dictionary, ["code", "name", "description"])
        dictionary.code = code.strip()
        dictionary.name = name.strip()
        dictionary.description = description.strip() or None
        try:
            changes = _diff_dict(before, _simple_snapshot(dictionary, ["code", "name", "description"]))
            if changes:
                _write_audit_log(
                    db,
                    action="update",
                    entity_type="dictionary",
                    entity_id=dictionary.id,
                    entity_label=dictionary.name,
                    actor=user,
                    changes=changes,
                )
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
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    dictionary = db.get(Dictionary, dictionary_id)
    if not dictionary:
        return _redirect("/dictionaries")

    used_by_attributes, used_in_categories, used_in_attribute_values = _dictionary_usage_counts(db, dictionary.id)
    if used_by_attributes or used_in_categories or used_in_attribute_values:
        reasons: list[str] = []
        if used_by_attributes:
            reasons.append(f"Р В РЎвЂ”Р РЋР вЂљР В РЎвЂР В Р вЂ Р РЋР РЏР В Р’В·Р В Р’В°Р В Р вЂ¦ Р В РЎвЂќ Р В Р’В°Р РЋРІР‚С™Р РЋР вЂљР В РЎвЂР В Р’В±Р РЋРЎвЂњР РЋРІР‚С™Р В Р’В°Р В РЎВ ({used_by_attributes})")
        if used_in_categories:
            reasons.append(f"Р В РЎвЂР РЋР С“Р В РЎвЂ”Р В РЎвЂўР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋРЎвЂњР В Р’ВµР РЋРІР‚С™Р РЋР С“Р РЋР РЏ Р В РЎвЂќР В Р’В°Р В РЎвЂќ Р В РЎвЂќР В Р’В°Р РЋРІР‚С™Р В Р’ВµР В РЎвЂ“Р В РЎвЂўР РЋР вЂљР В РЎвЂР РЋР РЏ Р В Р вЂ  Р РЋРІР‚С™Р В РЎвЂўР В Р вЂ Р В Р’В°Р РЋР вЂљР В Р’В°Р РЋРІР‚В¦ ({used_in_categories})")
        if used_in_attribute_values:
            reasons.append(f"Р В РЎвЂР РЋР С“Р В РЎвЂ”Р В РЎвЂўР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋРЎвЂњР В Р’ВµР РЋРІР‚С™Р РЋР С“Р РЋР РЏ Р В Р вЂ  Р В Р’В·Р В Р вЂ¦Р В Р’В°Р РЋРІР‚РЋР В Р’ВµР В Р вЂ¦Р В РЎвЂР РЋР РЏР РЋРІР‚В¦ Р В Р’В°Р РЋРІР‚С™Р РЋР вЂљР В РЎвЂР В Р’В±Р РЋРЎвЂњР РЋРІР‚С™Р В РЎвЂўР В Р вЂ  Р РЋРІР‚С™Р В РЎвЂўР В Р вЂ Р В Р’В°Р РЋР вЂљР В РЎвЂўР В Р вЂ  ({used_in_attribute_values})")
        _set_flash(request, f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р РЋР С“Р В РЎвЂ”Р РЋР вЂљР В Р’В°Р В Р вЂ Р В РЎвЂўР РЋРІР‚РЋР В Р вЂ¦Р В РЎвЂР В РЎвЂќ: {', '.join(reasons)}", "error")
        return _redirect(f"/dictionaries/{dictionary_id}")

    _write_audit_log(
        db,
        action="delete",
        entity_type="dictionary",
        entity_id=dictionary.id,
        entity_label=dictionary.name,
        actor=user,
        changes=_audit_delete_changes(_simple_snapshot(dictionary, ["code", "name", "description"])),
    )
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
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    auto_sort_order = _next_available_sort_order(db, dictionary_id)
    item = DictionaryItem(
        dictionary_id=dictionary_id,
        code=code.strip(),
        label=label.strip(),
        sort_order=auto_sort_order,
        is_active=active is not None,
    )
    db.add(item)
    try:
        db.flush()
        _write_audit_log(
            db,
            action="create",
            entity_type="dictionary_item",
            entity_id=item.id,
            entity_label=item.label,
            actor=user,
            changes=_audit_create_changes(_simple_snapshot(item, ["code", "label", "sort_order", "is_active"])),
            scope_type="dictionary",
            scope_id=dictionary_id,
        )
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
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    item = db.get(DictionaryItem, item_id)
    if item:
        before = _simple_snapshot(item, ["code", "label", "sort_order", "is_active"])
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
            changes = _diff_dict(before, _simple_snapshot(item, ["code", "label", "sort_order", "is_active"]))
            if changes:
                _write_audit_log(
                    db,
                    action="update",
                    entity_type="dictionary_item",
                    entity_id=item.id,
                    entity_label=item.label,
                    actor=user,
                    changes=changes,
                    scope_type="dictionary",
                    scope_id=item.dictionary_id,
                )
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
    user: User = Depends(require_roles("admin")),
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
            reasons.append(f"Р В РЎвЂќР В Р’В°Р РЋРІР‚С™Р В Р’ВµР В РЎвЂ“Р В РЎвЂўР РЋР вЂљР В РЎвЂР РЋР РЏ Р РЋРЎвЂњ Р РЋРІР‚С™Р В РЎвЂўР В Р вЂ Р В Р’В°Р РЋР вЂљР В РЎвЂўР В Р вЂ  ({used_as_category})")
        if used_in_attribute_values:
            reasons.append(f"Р В Р’В·Р В Р вЂ¦Р В Р’В°Р РЋРІР‚РЋР В Р’ВµР В Р вЂ¦Р В РЎвЂР РЋР РЏ Р В Р’В°Р РЋРІР‚С™Р РЋР вЂљР В РЎвЂР В Р’В±Р РЋРЎвЂњР РЋРІР‚С™Р В РЎвЂўР В Р вЂ  Р РЋРІР‚С™Р В РЎвЂўР В Р вЂ Р В Р’В°Р РЋР вЂљР В РЎвЂўР В Р вЂ  ({used_in_attribute_values})")
        _set_flash(request, f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р РЋР РЉР В Р’В»Р В Р’ВµР В РЎВР В Р’ВµР В Р вЂ¦Р РЋРІР‚С™: Р В РЎвЂР РЋР С“Р В РЎвЂ”Р В РЎвЂўР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋРЎвЂњР В Р’ВµР РЋРІР‚С™Р РЋР С“Р РЋР РЏ Р В Р вЂ  {', '.join(reasons)}", "error")
        return _redirect(f"/dictionaries/{item.dictionary_id}")

    dictionary_id = item.dictionary_id
    _write_audit_log(
        db,
        action="delete",
        entity_type="dictionary_item",
        entity_id=item.id,
        entity_label=item.label,
        actor=user,
        changes=_audit_delete_changes(_simple_snapshot(item, ["code", "label", "sort_order", "is_active"])),
        scope_type="dictionary",
        scope_id=dictionary_id,
    )
    db.delete(item)
    db.commit()
    _set_flash(request, "Элемент справочника удален", "success")
    return _redirect(f"/dictionaries/{dictionary_id}")


@app.get("/attributes", response_class=HTMLResponse)
def attributes_page(
    request: Request,
    q: str = "",
    data_type: str = "",
    group_code: str = "",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt: Select[tuple[Attribute]] = select(Attribute).options(joinedload(Attribute.dictionary))
    if q:
        stmt = stmt.where(or_(Attribute.code.ilike(f"%{q}%"), Attribute.name.ilike(f"%{q}%")))
    if data_type:
        stmt = stmt.where(Attribute.data_type == data_type)
    if group_code:
        stmt = stmt.where(Attribute.group_code == group_code)

    attributes = list(db.scalars(stmt.order_by(Attribute.name.asc())).all())
    dictionaries = list(db.scalars(select(Dictionary).order_by(Dictionary.name.asc())).all())
    return _render(
        request,
        "attributes/list.html",
        {
            "title": "Атрибуты",
            "attributes": attributes,
            "attribute_groups_view": _catalog_attribute_groups(attributes),
            "dictionaries": dictionaries,
            "data_types": sorted(DATA_TYPES),
            "attribute_groups": [(key, ATTRIBUTE_GROUP_LABELS.get(key, key)) for key in ATTRIBUTE_GROUP_ORDER],
            "q": q,
            "selected_type": data_type,
            "selected_group": group_code,
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
    group_code: str = Form("fashion_spec"),
    is_required: str | None = Form(None),
    is_multivalue: str | None = Form(None),
    dictionary_id_raw: str = Form(""),
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    dictionary_id: int | None = None
    if dictionary_id_raw.strip():
        try:
            dictionary_id = int(dictionary_id_raw)
        except ValueError:
            _set_flash(request, "Некорректный справочник", "error")
            return _redirect("/attributes")
        if not db.get(Dictionary, dictionary_id):
            _set_flash(request, "Справочник не найден", "error")
            return _redirect("/attributes")
    form_error = _attribute_form_error(
        db,
        code=code,
        name=name,
        data_type=data_type,
        group_code=group_code,
        is_multivalue=is_multivalue is not None,
        dictionary_id=dictionary_id if data_type == "enum" else None,
    )
    if form_error:
        _set_flash(request, form_error, "error")
        return _redirect("/attributes")

    attribute = Attribute(
        code=code.strip(),
        name=name.strip(),
        group_code=group_code,
        data_type=data_type,
        is_required=is_required is not None,
        is_multivalue=is_multivalue is not None,
        dictionary_id=dictionary_id if data_type == "enum" else None,
        is_active=True,
    )
    db.add(attribute)
    try:
        db.flush()
        _write_audit_log(
            db,
            action="create",
            entity_type="attribute",
            entity_id=attribute.id,
            entity_label=attribute.name,
            actor=user,
            changes=_audit_create_changes(
                _simple_snapshot(
                    attribute,
                    ["code", "name", "group_code", "data_type", "is_required", "is_multivalue", "dictionary_id", "is_active"],
                )
            ),
        )
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
            "attribute_groups": [(key, ATTRIBUTE_GROUP_LABELS.get(key, key)) for key in ATTRIBUTE_GROUP_ORDER],
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
    group_code: str = Form("fashion_spec"),
    is_required: str | None = Form(None),
    is_multivalue: str | None = Form(None),
    dictionary_id_raw: str = Form(""),
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    attribute = db.get(Attribute, attribute_id)
    if not attribute:
        return _redirect("/attributes")
    before = _simple_snapshot(
        attribute,
        ["code", "name", "group_code", "data_type", "is_required", "is_multivalue", "dictionary_id", "is_active"],
    )

    if attribute.data_type != data_type and not can_change_attribute_type(db, attribute.id):
        _set_flash(request, "Нельзя изменить тип: по атрибуту уже есть значения", "error")
        return _redirect(f"/attributes/{attribute_id}")
    dictionary_id: int | None = None
    if dictionary_id_raw.strip():
        try:
            dictionary_id = int(dictionary_id_raw)
        except ValueError:
            _set_flash(request, "Некорректный справочник", "error")
            return _redirect(f"/attributes/{attribute_id}")
        if not db.get(Dictionary, dictionary_id):
            _set_flash(request, "Справочник не найден", "error")
            return _redirect(f"/attributes/{attribute_id}")
    if data_type == "enum" and attribute.dictionary_id != dictionary_id and _attribute_has_values(db, attribute.id):
        _set_flash(request, "Нельзя менять справочник: по атрибуту уже есть значения", "error")
        return _redirect(f"/attributes/{attribute_id}")
    form_error = _attribute_form_error(
        db,
        code=code,
        name=name,
        data_type=data_type,
        group_code=group_code,
        is_multivalue=is_multivalue is not None,
        dictionary_id=dictionary_id if data_type == "enum" else None,
        attribute_id=attribute.id,
    )
    if form_error:
        _set_flash(request, form_error, "error")
        return _redirect(f"/attributes/{attribute_id}")

    attribute.code = code.strip()
    attribute.name = name.strip()
    attribute.group_code = group_code
    attribute.data_type = data_type
    attribute.is_required = is_required is not None
    attribute.is_multivalue = is_multivalue is not None
    attribute.dictionary_id = dictionary_id if data_type == "enum" else None

    try:
        changes = _diff_dict(
            before,
            _simple_snapshot(
                attribute,
                ["code", "name", "group_code", "data_type", "is_required", "is_multivalue", "dictionary_id", "is_active"],
            ),
        )
        if changes:
            _write_audit_log(
                db,
                action="update",
                entity_type="attribute",
                entity_id=attribute.id,
                entity_label=attribute.name,
                actor=user,
                changes=changes,
            )
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
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    attribute = db.get(Attribute, attribute_id)
    if attribute:
        before_is_active = attribute.is_active
        attribute.is_active = False
        _write_audit_log(
            db,
            action="update",
            entity_type="attribute",
            entity_id=attribute.id,
            entity_label=attribute.name,
            actor=user,
            changes={"is_active": _field_change(before_is_active, False)},
            summary=f"Атрибут деактивирован: {attribute.name}",
        )
        db.commit()
        _set_flash(request, "Атрибут деактивирован", "success")
    return _redirect("/attributes")


@app.post("/attributes/{attribute_id}/delete")
def delete_attribute(
    attribute_id: int,
    request: Request,
    user: User = Depends(require_roles("admin")),
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
            f"Р В РЎСљР В Р’ВµР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋР РЏ Р РЋРЎвЂњР В РўвЂР В Р’В°Р В Р’В»Р В РЎвЂР РЋРІР‚С™Р РЋР Р‰ Р В Р’В°Р РЋРІР‚С™Р РЋР вЂљР В РЎвЂР В Р’В±Р РЋРЎвЂњР РЋРІР‚С™: Р В РЎвЂўР В Р вЂ¦ Р В РЎвЂР РЋР С“Р В РЎвЂ”Р В РЎвЂўР В Р’В»Р РЋР Р‰Р В Р’В·Р РЋРЎвЂњР В Р’ВµР РЋРІР‚С™Р РЋР С“Р РЋР РЏ Р В Р вЂ  Р В РЎвЂР В Р’В·Р В РўвЂР В Р’ВµР В Р’В»Р В РЎвЂР РЋР РЏР РЋРІР‚В¦ ({used_in_products}). Р В Р Р‹Р В Р вЂ¦Р В Р’В°Р РЋРІР‚РЋР В Р’В°Р В Р’В»Р В Р’В° Р РЋРЎвЂњР В Р’В±Р В Р’ВµР РЋР вЂљР В РЎвЂР РЋРІР‚С™Р В Р’Вµ Р В Р’ВµР В РЎвЂ“Р В РЎвЂў Р В РЎвЂР В Р’В· Р В РЎвЂќР В Р’В°Р РЋР вЂљР РЋРІР‚С™Р В РЎвЂўР РЋРІР‚РЋР В Р’ВµР В РЎвЂќ.",
            "error",
        )
        return _redirect(f"/attributes/{attribute_id}")

    _write_audit_log(
        db,
        action="delete",
        entity_type="attribute",
        entity_id=attribute.id,
        entity_label=attribute.name,
        actor=user,
        changes=_audit_delete_changes(
            _simple_snapshot(
                attribute,
                ["code", "name", "data_type", "is_required", "is_multivalue", "dictionary_id", "is_active"],
            )
        ),
    )
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
        db.flush()
        _write_audit_log(
            db,
            action="create",
            entity_type="product",
            entity_id=product.id,
            entity_label=_product_label(product),
            actor=user,
            changes=_audit_create_changes(_product_snapshot(product)),
        )
        db.commit()
        if product.status == "active":
            errors = validate_product_completeness(db, product)
            if errors:
                before_status = product.status
                product.status = "draft"
                _write_audit_log(
                    db,
                    action="status_change",
                    entity_type="product",
                    entity_id=product.id,
                    entity_label=_product_label(product),
                    actor=user,
                    changes={"status": _field_change(before_status, "draft")},
                    summary=f"Статус изделия скорректирован после проверки: {_product_label(product)}",
                )
                db.commit()
                _set_flash(
                    request,
                    "Р В Р’ВР В Р’В·Р В РўвЂР В Р’ВµР В Р’В»Р В РЎвЂР В Р’Вµ Р РЋР С“Р В РЎвЂўР В Р’В·Р В РўвЂР В Р’В°Р В Р вЂ¦Р В РЎвЂў Р В РЎвЂќР В Р’В°Р В РЎвЂќ draft: Р В РўвЂР В Р’В»Р РЋР РЏ Р В Р’В°Р В РЎвЂќР РЋРІР‚С™Р В РЎвЂР В Р вЂ Р В Р’В°Р РЋРІР‚В Р В РЎвЂР В РЎвЂ Р В Р’В·Р В Р’В°Р В РЎвЂ”Р В РЎвЂўР В Р’В»Р В Р вЂ¦Р В РЎвЂР РЋРІР‚С™Р В Р’Вµ Р В РЎвЂўР В Р’В±Р РЋР РЏР В Р’В·Р В Р’В°Р РЋРІР‚С™Р В Р’ВµР В Р’В»Р РЋР Р‰Р В Р вЂ¦Р РЋРІР‚в„–Р В Р’Вµ Р В Р’В°Р РЋРІР‚С™Р РЋР вЂљР В РЎвЂР В Р’В±Р РЋРЎвЂњР РЋРІР‚С™Р РЋРІР‚в„–",
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
    attribute_groups_view = _attribute_groups(attribute_assignments)
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
    comments = _comment_rows(db, "product", product_id)
    plm_settings = _get_plm_settings(db)
    fashion_settings = _get_fashion_settings(db)
    product_materials = list(
        db.scalars(
            select(ProductMaterial)
            .options(joinedload(ProductMaterial.supplier))
            .where(ProductMaterial.product_id == product_id)
            .order_by(ProductMaterial.sort_order.asc(), ProductMaterial.id.asc())
        ).all()
    )
    product_bom_items = list(
        db.scalars(
            select(ProductBOMItem)
            .options(joinedload(ProductBOMItem.material))
            .where(ProductBOMItem.product_id == product_id)
            .order_by(ProductBOMItem.sort_order.asc(), ProductBOMItem.id.asc())
        ).all()
    )
    product_variants = list(
        db.scalars(
            select(ProductVariant)
            .where(ProductVariant.product_id == product_id)
            .order_by(ProductVariant.color.asc(), ProductVariant.size.asc())
        ).all()
    )
    product_samples = list(
        db.scalars(
            select(ProductSample)
            .options(joinedload(ProductSample.owner))
            .where(ProductSample.product_id == product_id)
            .order_by(ProductSample.id.desc())
        ).all()
    )
    product_costing_items = list(
        db.scalars(
            select(ProductCostingItem)
            .where(ProductCostingItem.product_id == product_id)
            .order_by(ProductCostingItem.sort_order.asc(), ProductCostingItem.id.asc())
        ).all()
    )
    product_costing_total = float(sum(float(item.amount or 0) for item in product_costing_items))
    product_audit_rows = _scope_audit_rows(db, "product", product_id, 30)
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
            "attribute_groups_view": attribute_groups_view,
            "enum_options_by_attr": enum_options_by_attr,
            "product_tasks": product_tasks,
            "comments": comments,
            "plm_settings": plm_settings,
            "fashion_settings": fashion_settings,
            "product_materials": product_materials,
            "product_bom_items": product_bom_items,
            "product_variants": product_variants,
            "product_samples": product_samples,
            "product_costing_items": product_costing_items,
            "product_costing_total": product_costing_total,
            "product_audit_rows": product_audit_rows,
            "get_value_view": get_value_view,
            "can_manage": user.role in {"admin", "user"},
            "user": user,
        },
    )


@app.post("/products/{product_id}/materials")
def add_product_material(
    product_id: int,
    request: Request,
    code: str = Form(""),
    name: str = Form(...),
    composition: str = Form(""),
    supplier_id_raw: str = Form(""),
    unit: str = Form(""),
    price_per_unit_raw: str = Form(""),
    currency: str = Form("RUB"),
    notes: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    try:
        supplier_id = int(supplier_id_raw) if supplier_id_raw.strip() else None
        price_per_unit = float(price_per_unit_raw.replace(",", ".")) if price_per_unit_raw.strip() else None
    except ValueError:
        _set_flash(request, "Проверьте числовые значения материала", "error")
        return _redirect(f"/products/{product_id}")
    material = ProductMaterial(
        product_id=product_id,
        supplier_id=supplier_id,
        code=code.strip() or None,
        name=name.strip(),
        composition=composition.strip() or None,
        unit=unit.strip() or None,
        price_per_unit=price_per_unit,
        currency=currency.strip() or "RUB",
        notes=notes.strip() or None,
        sort_order=int(db.scalar(select(func.count(ProductMaterial.id)).where(ProductMaterial.product_id == product_id)) or 0) + 1,
    )
    db.add(material)
    db.flush()
    _write_audit_log(
        db,
        action="create",
        entity_type="product_material",
        entity_id=material.id,
        entity_label=material.name,
        actor=user,
        changes=_audit_create_changes(_simple_snapshot(material, ["code", "name", "composition", "supplier_id", "unit", "price_per_unit", "currency"])),
        scope_type="product",
        scope_id=product_id,
    )
    _notify_users(
        db,
        _product_watcher_ids(product),
        event_key="plm_updated",
        title="PLM-блок: материалы",
        message=f"{user.full_name or user.login} добавил материал '{material.name}' в изделие {_product_label(product)}.",
        actor_id=user.id,
        link_url=f"/products/{product_id}",
        entity_type="product",
        entity_id=product_id,
    )
    db.commit()
    _set_flash(request, "Материал добавлен", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/materials/{material_id}/delete")
def delete_product_material(
    product_id: int,
    material_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    material = db.scalar(select(ProductMaterial).where(ProductMaterial.id == material_id, ProductMaterial.product_id == product_id))
    product = db.get(Product, product_id)
    if material:
        _write_audit_log(
            db,
            action="delete",
            entity_type="product_material",
            entity_id=material.id,
            entity_label=material.name,
            actor=user,
            changes=_audit_delete_changes(_simple_snapshot(material, ["code", "name", "composition", "supplier_id", "unit", "price_per_unit", "currency"])),
            scope_type="product",
            scope_id=product_id,
        )
        db.delete(material)
        _notify_users(
            db,
            _product_watcher_ids(product),
            event_key="plm_updated",
            title="PLM-блок: материалы",
            message=f"{user.full_name or user.login} удалил материал из изделия {_product_label(product)}.",
            actor_id=user.id,
            link_url=f"/products/{product_id}",
            entity_type="product",
            entity_id=product_id,
        )
        db.commit()
        _set_flash(request, "Материал удален", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/bom")
def add_product_bom_item(
    product_id: int,
    request: Request,
    component: str = Form(...),
    material_id_raw: str = Form(""),
    quantity_raw: str = Form(""),
    unit: str = Form(""),
    waste_percent_raw: str = Form(""),
    notes: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    try:
        material_id = int(material_id_raw) if material_id_raw.strip() else None
        quantity = float(quantity_raw.replace(",", ".")) if quantity_raw.strip() else None
        waste_percent = float(waste_percent_raw.replace(",", ".")) if waste_percent_raw.strip() else None
    except ValueError:
        _set_flash(request, "Проверьте числовые поля BOM", "error")
        return _redirect(f"/products/{product_id}")
    item = ProductBOMItem(
        product_id=product_id,
        material_id=material_id,
        component=component.strip(),
        quantity=quantity,
        unit=unit.strip() or None,
        waste_percent=waste_percent,
        notes=notes.strip() or None,
        sort_order=int(db.scalar(select(func.count(ProductBOMItem.id)).where(ProductBOMItem.product_id == product_id)) or 0) + 1,
    )
    db.add(item)
    db.flush()
    _write_audit_log(
        db,
        action="create",
        entity_type="product_bom",
        entity_id=item.id,
        entity_label=item.component,
        actor=user,
        changes=_audit_create_changes(_simple_snapshot(item, ["component", "material_id", "quantity", "unit", "waste_percent"])),
        scope_type="product",
        scope_id=product_id,
    )
    _notify_users(
        db,
        _product_watcher_ids(product),
        event_key="plm_updated",
        title="PLM-блок: BOM",
        message=f"{user.full_name or user.login} добавил BOM-компонент '{item.component}' в изделие {_product_label(product)}.",
        actor_id=user.id,
        link_url=f"/products/{product_id}",
        entity_type="product",
        entity_id=product_id,
    )
    db.commit()
    _set_flash(request, "Строка BOM добавлена", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/bom/{item_id}/delete")
def delete_product_bom_item(
    product_id: int,
    item_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    item = db.scalar(select(ProductBOMItem).where(ProductBOMItem.id == item_id, ProductBOMItem.product_id == product_id))
    product = db.get(Product, product_id)
    if item:
        _write_audit_log(
            db,
            action="delete",
            entity_type="product_bom",
            entity_id=item.id,
            entity_label=item.component,
            actor=user,
            changes=_audit_delete_changes(_simple_snapshot(item, ["component", "material_id", "quantity", "unit", "waste_percent"])),
            scope_type="product",
            scope_id=product_id,
        )
        db.delete(item)
        _notify_users(
            db,
            _product_watcher_ids(product),
            event_key="plm_updated",
            title="PLM-блок: BOM",
            message=f"{user.full_name or user.login} удалил строку BOM из изделия {_product_label(product)}.",
            actor_id=user.id,
            link_url=f"/products/{product_id}",
            entity_type="product",
            entity_id=product_id,
        )
        db.commit()
        _set_flash(request, "Строка BOM удалена", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/variants")
def add_product_variant(
    product_id: int,
    request: Request,
    color: str = Form(...),
    size: str = Form(...),
    sku_suffix: str = Form(""),
    ean: str = Form(""),
    planned_qty_raw: str = Form(""),
    status: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    try:
        planned_qty = int(planned_qty_raw) if planned_qty_raw.strip() else None
    except ValueError:
        _set_flash(request, "Плановое количество должно быть числом", "error")
        return _redirect(f"/products/{product_id}")
    variant = ProductVariant(
        product_id=product_id,
        color=color.strip(),
        size=size.strip(),
        sku_suffix=sku_suffix.strip() or None,
        ean=ean.strip() or None,
        planned_qty=planned_qty,
        status=status.strip() or None,
    )
    db.add(variant)
    db.flush()
    _write_audit_log(
        db,
        action="create",
        entity_type="product_variant",
        entity_id=variant.id,
        entity_label=f"{variant.color}/{variant.size}",
        actor=user,
        changes=_audit_create_changes(_simple_snapshot(variant, ["color", "size", "sku_suffix", "ean", "planned_qty", "status"])),
        scope_type="product",
        scope_id=product_id,
    )
    _notify_users(
        db,
        _product_watcher_ids(product),
        event_key="plm_updated",
        title="PLM-блок: матрица",
        message=f"{user.full_name or user.login} добавил вариант {variant.color}/{variant.size} в изделие {_product_label(product)}.",
        actor_id=user.id,
        link_url=f"/products/{product_id}",
        entity_type="product",
        entity_id=product_id,
    )
    db.commit()
    _set_flash(request, "Вариант добавлен", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/variants/{variant_id}/delete")
def delete_product_variant(
    product_id: int,
    variant_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    variant = db.scalar(select(ProductVariant).where(ProductVariant.id == variant_id, ProductVariant.product_id == product_id))
    product = db.get(Product, product_id)
    if variant:
        _write_audit_log(
            db,
            action="delete",
            entity_type="product_variant",
            entity_id=variant.id,
            entity_label=f"{variant.color}/{variant.size}",
            actor=user,
            changes=_audit_delete_changes(_simple_snapshot(variant, ["color", "size", "sku_suffix", "ean", "planned_qty", "status"])),
            scope_type="product",
            scope_id=product_id,
        )
        db.delete(variant)
        _notify_users(
            db,
            _product_watcher_ids(product),
            event_key="plm_updated",
            title="PLM-блок: матрица",
            message=f"{user.full_name or user.login} удалил вариант из изделия {_product_label(product)}.",
            actor_id=user.id,
            link_url=f"/products/{product_id}",
            entity_type="product",
            entity_id=product_id,
        )
        db.commit()
        _set_flash(request, "Вариант удален", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/samples")
def add_product_sample(
    product_id: int,
    request: Request,
    sample_type: str = Form(...),
    status: str = Form(...),
    owner_id_raw: str = Form(""),
    due_date: str = Form(""),
    received_date: str = Form(""),
    notes: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    try:
        owner_id = int(owner_id_raw) if owner_id_raw.strip() else None
        parsed_due_date = datetime.fromisoformat(due_date).date() if due_date.strip() else None
        parsed_received_date = datetime.fromisoformat(received_date).date() if received_date.strip() else None
    except ValueError:
        _set_flash(request, "Проверьте даты трекинга образца", "error")
        return _redirect(f"/products/{product_id}")
    sample = ProductSample(
        product_id=product_id,
        sample_type=sample_type.strip(),
        status=status.strip(),
        owner_id=owner_id,
        due_date=parsed_due_date,
        received_date=parsed_received_date,
        notes=notes.strip() or None,
    )
    db.add(sample)
    db.flush()
    _write_audit_log(
        db,
        action="create",
        entity_type="product_sample",
        entity_id=sample.id,
        entity_label=sample.sample_type,
        actor=user,
        changes=_audit_create_changes(_simple_snapshot(sample, ["sample_type", "status", "owner_id", "due_date", "received_date"])),
        scope_type="product",
        scope_id=product_id,
    )
    notify_ids = _product_watcher_ids(product)
    if sample.owner_id:
        notify_ids.add(sample.owner_id)
    _notify_users(
        db,
        notify_ids,
        event_key="plm_updated",
        title="PLM-блок: sample tracking",
        message=f"{user.full_name or user.login} добавил образец '{sample.sample_type}' для изделия {_product_label(product)}.",
        actor_id=user.id,
        link_url=f"/products/{product_id}",
        entity_type="product",
        entity_id=product_id,
    )
    db.commit()
    _set_flash(request, "Образец добавлен", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/samples/{sample_id}/delete")
def delete_product_sample(
    product_id: int,
    sample_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    sample = db.scalar(select(ProductSample).where(ProductSample.id == sample_id, ProductSample.product_id == product_id))
    product = db.get(Product, product_id)
    if sample:
        notify_ids = _product_watcher_ids(product)
        if sample.owner_id:
            notify_ids.add(sample.owner_id)
        _write_audit_log(
            db,
            action="delete",
            entity_type="product_sample",
            entity_id=sample.id,
            entity_label=sample.sample_type,
            actor=user,
            changes=_audit_delete_changes(_simple_snapshot(sample, ["sample_type", "status", "owner_id", "due_date", "received_date"])),
            scope_type="product",
            scope_id=product_id,
        )
        db.delete(sample)
        _notify_users(
            db,
            notify_ids,
            event_key="plm_updated",
            title="PLM-блок: sample tracking",
            message=f"{user.full_name or user.login} удалил запись sample tracking из изделия {_product_label(product)}.",
            actor_id=user.id,
            link_url=f"/products/{product_id}",
            entity_type="product",
            entity_id=product_id,
        )
        db.commit()
        _set_flash(request, "Образец удален", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/costing")
def add_product_costing_item(
    product_id: int,
    request: Request,
    cost_group: str = Form(...),
    label: str = Form(...),
    amount_raw: str = Form(...),
    currency: str = Form("RUB"),
    notes: str = Form(""),
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    product = db.get(Product, product_id)
    if not product:
        return _redirect("/products")
    try:
        amount = float(amount_raw.replace(",", "."))
    except ValueError:
        _set_flash(request, "Сумма должна быть числом", "error")
        return _redirect(f"/products/{product_id}")
    item = ProductCostingItem(
        product_id=product_id,
        cost_group=cost_group.strip(),
        label=label.strip(),
        amount=amount,
        currency=currency.strip() or "RUB",
        notes=notes.strip() or None,
        sort_order=int(db.scalar(select(func.count(ProductCostingItem.id)).where(ProductCostingItem.product_id == product_id)) or 0) + 1,
    )
    db.add(item)
    db.flush()
    _write_audit_log(
        db,
        action="create",
        entity_type="product_costing",
        entity_id=item.id,
        entity_label=item.label,
        actor=user,
        changes=_audit_create_changes(_simple_snapshot(item, ["cost_group", "label", "amount", "currency"])),
        scope_type="product",
        scope_id=product_id,
    )
    _notify_users(
        db,
        _product_watcher_ids(product),
        event_key="plm_updated",
        title="PLM-блок: costing",
        message=f"{user.full_name or user.login} добавил статью cost decomposition в изделие {_product_label(product)}.",
        actor_id=user.id,
        link_url=f"/products/{product_id}",
        entity_type="product",
        entity_id=product_id,
    )
    db.commit()
    _set_flash(request, "Статья калькуляции добавлена", "success")
    return _redirect(f"/products/{product_id}")


@app.post("/products/{product_id}/costing/{item_id}/delete")
def delete_product_costing_item(
    product_id: int,
    item_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    item = db.scalar(select(ProductCostingItem).where(ProductCostingItem.id == item_id, ProductCostingItem.product_id == product_id))
    product = db.get(Product, product_id)
    if item:
        _write_audit_log(
            db,
            action="delete",
            entity_type="product_costing",
            entity_id=item.id,
            entity_label=item.label,
            actor=user,
            changes=_audit_delete_changes(_simple_snapshot(item, ["cost_group", "label", "amount", "currency"])),
            scope_type="product",
            scope_id=product_id,
        )
        db.delete(item)
        _notify_users(
            db,
            _product_watcher_ids(product),
            event_key="plm_updated",
            title="PLM-блок: costing",
            message=f"{user.full_name or user.login} удалил статью costing decomposition из изделия {_product_label(product)}.",
            actor_id=user.id,
            link_url=f"/products/{product_id}",
            entity_type="product",
            entity_id=product_id,
        )
        db.commit()
        _set_flash(request, "Статья калькуляции удалена", "success")
    return _redirect(f"/products/{product_id}")


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
    before = _product_snapshot(product)
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
        changes = _diff_dict(before, _product_snapshot(product))
        if changes:
            _write_audit_log(
                db,
                action="update",
                entity_type="product",
                entity_id=product.id,
                entity_label=_product_label(product),
                actor=user,
                changes=changes,
            )
            _notify_users(
                db,
                _product_watcher_ids(product),
                event_key="product_updated",
                title="Изделие обновлено",
                message=f"{user.full_name or user.login} обновил карточку изделия {_product_label(product)}.",
                actor_id=user.id,
                link_url=f"/products/{product.id}",
                entity_type="product",
                entity_id=product.id,
            )
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
    before_cover = product.cover_image_path

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
    _write_audit_log(
        db,
        action="update",
        entity_type="product",
        entity_id=product.id,
        entity_label=_product_label(product),
        actor=user,
        changes={"cover_image_path": _field_change(before_cover, product.cover_image_path)},
        summary=f"Обложка изделия обновлена: {_product_label(product)}",
    )
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
    db.flush()
    _write_audit_log(
        db,
        action="upload",
        entity_type="product_file",
        entity_id=rec.id,
        entity_label=rec.original_name,
        actor=user,
        changes=_audit_create_changes(
            _simple_snapshot(rec, ["category", "title", "original_name", "file_path", "mime_type", "uploaded_by"])
        ),
        scope_type="product",
        scope_id=product_id,
    )
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
    snapshot = _simple_snapshot(rec, ["category", "title", "original_name", "file_path", "mime_type", "uploaded_by"])

    if rec.file_path.startswith("/static/product_files/"):
        file_name = rec.file_path.split("/static/product_files/", 1)[1]
        file_path = PRODUCT_FILES_DIR / file_name
        if file_path.exists():
            file_path.unlink()

    product = db.get(Product, product_id)
    _write_audit_log(
        db,
        action="remove",
        entity_type="product_file",
        entity_id=rec.id,
        entity_label=rec.original_name,
        actor=user,
        changes=_audit_delete_changes(snapshot),
        scope_type="product",
        scope_id=product_id,
    )
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
    before_spec = _product_spec_snapshot(spec)

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

    assignments = _load_product_attribute_assignments(db, product_id)
    before_attr = _product_attributes_snapshot(assignments)
    assignment_errors: list[str] = []
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
    db.flush()
    spec_changes = _diff_dict(before_spec, _product_spec_snapshot(spec))
    attribute_changes = _diff_dict(before_attr, _product_attributes_snapshot(assignments))
    changes = {**spec_changes, **attribute_changes}
    if changes:
        _write_audit_log(
            db,
            action="update",
            entity_type="product_spec",
            entity_id=spec.id,
            entity_label=_product_label(product),
            actor=user,
            changes=changes,
            summary=f"Fashion-спецификация обновлена: {_product_label(product)}",
            scope_type="product",
            scope_id=product_id,
        )
        _notify_users(
            db,
            _product_watcher_ids(product),
            event_key="product_updated",
            title="Fashion-спецификация обновлена",
            message=f"{user.full_name or user.login} обновил fashion-спецификацию изделия {_product_label(product)}.",
            actor_id=user.id,
            link_url=f"/products/{product.id}",
            entity_type="product",
            entity_id=product.id,
        )
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
    before = _product_team_snapshot(product)

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
    changes = _diff_dict(before, _product_team_snapshot(product))
    if changes:
        _write_audit_log(
            db,
            action="update",
            entity_type="product_team",
            entity_id=product.id,
            entity_label=_product_label(product),
            actor=user,
            changes=changes,
            summary=f"Команда изделия обновлена: {_product_label(product)}",
            scope_type="product",
            scope_id=product_id,
        )
        notify_ids = _product_watcher_ids(product)
        notify_ids |= {uid for uid in [designer_id, product_manager_id, pattern_maker_id, technologist_id, department_head_id] if uid}
        _notify_users(
            db,
            notify_ids,
            event_key="product_team_assigned",
            title="Команда изделия обновлена",
            message=f"{user.full_name or user.login} обновил команду изделия {_product_label(product)}.",
            actor_id=user.id,
            link_url=f"/products/{product.id}",
            entity_type="product",
            entity_id=product.id,
        )
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
        before_status = product.status
        product.status = "archived"
        product.updated_by = user.id
        product.updated_at = datetime.utcnow()
        _write_audit_log(
            db,
            action="archive",
            entity_type="product",
            entity_id=product.id,
            entity_label=_product_label(product),
            actor=user,
            changes={"status": _field_change(before_status, "archived")},
        )
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

    before_status = product.status
    completeness_errors = validate_product_completeness(db, product)
    product.status = "active" if not completeness_errors else "draft"
    product.updated_by = user.id
    product.updated_at = datetime.utcnow()
    _write_audit_log(
        db,
        action="restore",
        entity_type="product",
        entity_id=product.id,
        entity_label=_product_label(product),
        actor=user,
        changes={"status": _field_change(before_status, product.status)},
    )
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
            .order_by(Attribute.group_code.asc(), Attribute.name.asc())
        ).all()
    )

    return _render(
        request,
        "products/attributes.html",
        {
            "title": f"Атрибуты изделия: {product.name}",
            "product": product,
            "assignments": assignments,
            "assignment_groups": _attribute_groups(assignments),
            "available_attributes": available_attributes,
            "available_attribute_groups": _catalog_attribute_groups(available_attributes),
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
    user: User = Depends(require_roles("admin", "user")),
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
    db.flush()
    _write_audit_log(
        db,
        action="create",
        entity_type="product_attribute",
        entity_id=assignment.id,
        entity_label=attribute.name,
        actor=user,
        summary=f"Атрибут добавлен к изделию: {attribute.name}",
        scope_type="product",
        scope_id=product_id,
    )
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
            "title": f"Редактирование атрибута: {assignment.attribute.name}",
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
        .options(
            joinedload(ProductAttributeAssignment.attribute),
            selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
        )
        .where(ProductAttributeAssignment.id == assignment_id, ProductAttributeAssignment.product_id == product_id)
    )
    if not assignment:
        return _redirect(f"/products/{product_id}/attributes")

    attr = assignment.attribute
    before_value = _assignment_value_snapshot(assignment)
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
    after_value = _assignment_value_snapshot(assignment)
    if before_value != after_value:
        _write_audit_log(
            db,
            action="update",
            entity_type="product_attribute",
            entity_id=assignment.id,
            entity_label=attr.name,
            actor=user,
            changes={attr.name: _field_change(before_value, after_value)},
            summary=f"Атрибут изделия обновлен: {attr.name}",
            scope_type="product",
            scope_id=product_id,
        )
    db.commit()
    _set_flash(request, "Значение атрибута обновлено", "success")
    return _redirect(f"/products/{product_id}/attributes")


@app.post("/products/{product_id}/attributes/{assignment_id}/remove")
def remove_product_attribute(
    product_id: int,
    assignment_id: int,
    request: Request,
    user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    assignment = db.scalar(
        select(ProductAttributeAssignment)
        .options(
            joinedload(ProductAttributeAssignment.attribute),
            selectinload(ProductAttributeAssignment.values).joinedload(ProductAttributeValue.dictionary_item),
        )
        .where(
            ProductAttributeAssignment.id == assignment_id,
            ProductAttributeAssignment.product_id == product_id,
        )
    )
    if assignment:
        if assignment.attribute and assignment.attribute.is_required:
            _set_flash(request, "Нельзя снять обязательный атрибут с изделия", "error")
            return _redirect(f"/products/{product_id}/attributes")
        before_value = _assignment_value_snapshot(assignment)
        entity_label = assignment.attribute.name if assignment.attribute else f"Атрибут #{assignment_id}"
        _write_audit_log(
            db,
            action="delete",
            entity_type="product_attribute",
            entity_id=assignment.id,
            entity_label=entity_label,
            actor=user,
            changes={entity_label: _field_change(before_value, None)},
            summary=f"Атрибут снят с изделия: {entity_label}",
            scope_type="product",
            scope_id=product_id,
        )
        db.delete(assignment)
        db.commit()
        _set_flash(request, "Атрибут снят с изделия", "success")
    return _redirect(f"/products/{product_id}/attributes")







