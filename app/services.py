from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from .models import (
    Attribute,
    Dictionary,
    DictionaryItem,
    Product,
    ProductAttributeAssignment,
    ProductAttributeValue,
)

DATA_TYPES = {"string", "number", "date", "bool", "enum"}
ROLES = {"admin", "content-manager", "dictionary-manager", "read-only"}
PRODUCT_STATUSES = {"draft", "active", "archived"}


def get_category_items(db: Session) -> list[DictionaryItem]:
    category_dict = db.scalar(select(Dictionary).where(Dictionary.code == "category"))
    if not category_dict:
        return []
    stmt: Select[tuple[DictionaryItem]] = (
        select(DictionaryItem)
        .where(DictionaryItem.dictionary_id == category_dict.id, DictionaryItem.is_active.is_(True))
        .order_by(DictionaryItem.sort_order.asc(), DictionaryItem.label.asc())
    )
    return list(db.scalars(stmt).all())


def parse_multiline(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def clear_values(assignment: ProductAttributeAssignment) -> None:
    assignment.values.clear()


def validate_and_set_values(
    db: Session,
    assignment: ProductAttributeAssignment,
    attribute: Attribute,
    payload: str | bool | list[str] | None,
) -> list[str]:
    errors: list[str] = []
    clear_values(assignment)

    if attribute.data_type == "bool":
        values = [bool(payload)] if payload is not None else []
    elif isinstance(payload, list):
        values = payload
    elif isinstance(payload, str):
        values = parse_multiline(payload) if attribute.is_multivalue else [payload.strip()] if payload.strip() else []
    else:
        values = []

    if attribute.is_required and not values:
        return [f"Атрибут '{attribute.name}' обязателен."]

    if not attribute.is_multivalue and len(values) > 1:
        return [f"Атрибут '{attribute.name}' не допускает множественные значения."]

    for value in values:
        pav = ProductAttributeValue()
        if attribute.data_type == "string":
            pav.value_string = str(value)
        elif attribute.data_type == "number":
            try:
                pav.value_number = Decimal(str(value))
            except (InvalidOperation, ValueError):
                errors.append(f"Атрибут '{attribute.name}': некорректное число '{value}'.")
                continue
        elif attribute.data_type == "date":
            try:
                pav.value_date = date.fromisoformat(str(value))
            except ValueError:
                errors.append(f"Атрибут '{attribute.name}': используйте дату в формате YYYY-MM-DD.")
                continue
        elif attribute.data_type == "bool":
            pav.value_bool = bool(value)
        elif attribute.data_type == "enum":
            try:
                item_id = int(value)
            except (TypeError, ValueError):
                errors.append(f"Атрибут '{attribute.name}': некорректное значение справочника.")
                continue
            item = db.get(DictionaryItem, item_id)
            if not item:
                errors.append(f"Атрибут '{attribute.name}': элемент справочника не найден.")
                continue
            if attribute.dictionary_id and item.dictionary_id != attribute.dictionary_id:
                errors.append(f"Атрибут '{attribute.name}': выбран элемент чужого справочника.")
                continue
            if not item.is_active:
                errors.append(f"Атрибут '{attribute.name}': выбран неактивный элемент справочника.")
                continue
            pav.dictionary_item_id = item.id
        else:
            errors.append(f"Атрибут '{attribute.name}': неизвестный тип данных.")
            continue

        assignment.values.append(pav)

    if attribute.is_required and not assignment.values:
        errors.append(f"Атрибут '{attribute.name}' обязателен.")

    return errors


def get_value_view(value: ProductAttributeValue, data_type: str) -> str:
    if data_type == "string":
        return value.value_string or ""
    if data_type == "number":
        return "" if value.value_number is None else str(value.value_number)
    if data_type == "date":
        return "" if value.value_date is None else value.value_date.isoformat()
    if data_type == "bool":
        return "Да" if value.value_bool else "Нет"
    if data_type == "enum":
        return value.dictionary_item.label if value.dictionary_item else ""
    return ""


def validate_product_completeness(db: Session, product: Product) -> list[str]:
    errors: list[str] = []
    required_attributes = list(
        db.scalars(select(Attribute).where(Attribute.is_required.is_(True), Attribute.is_active.is_(True))).all()
    )
    assignments = {
        a.attribute_id: a
        for a in db.scalars(
            select(ProductAttributeAssignment).where(ProductAttributeAssignment.product_id == product.id)
        ).all()
    }

    for attribute in required_attributes:
        assignment = assignments.get(attribute.id)
        if not assignment or not assignment.values:
            errors.append(f"Не заполнен обязательный атрибут '{attribute.name}'.")

    return errors


def can_change_attribute_type(db: Session, attribute_id: int) -> bool:
    stmt = (
        select(ProductAttributeValue.id)
        .join(ProductAttributeAssignment, ProductAttributeAssignment.id == ProductAttributeValue.assignment_id)
        .where(ProductAttributeAssignment.attribute_id == attribute_id)
        .limit(1)
    )
    return db.execute(stmt).first() is None
