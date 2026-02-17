from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from app.config import DB_PATH
from app.database import Base, SessionLocal, engine
from app.models import (
    Attribute,
    Collection,
    Dictionary,
    DictionaryItem,
    Product,
    ProductAttributeAssignment,
    ProductAttributeValue,
    Task,
    TaskBoard,
    TaskQueue,
    ProductSpec,
    Supplier,
    User,
)
from app.security import hash_password


def reset_db() -> None:
    # Avoid OS-level file locking issues on Windows: recreate schema in place.
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def seed() -> None:
    db = SessionLocal()
    try:
        users = [
            User(login="admin", password_hash=hash_password("admin"), role="admin", is_active=True, created_at=datetime.utcnow()),
            User(
                login="content",
                password_hash=hash_password("content"),
                role="content-manager",
                is_active=True,
                created_at=datetime.utcnow() - timedelta(days=1),
            ),
            User(
                login="dict",
                password_hash=hash_password("dict"),
                role="dictionary-manager",
                is_active=True,
                created_at=datetime.utcnow() - timedelta(days=2),
            ),
            User(
                login="viewer",
                password_hash=hash_password("viewer"),
                role="read-only",
                is_active=True,
                created_at=datetime.utcnow() - timedelta(days=3),
            ),
        ]
        extra_roles = ["content-manager", "dictionary-manager", "read-only", "content-manager", "read-only", "admin"]
        for i, role in enumerate(extra_roles, start=5):
            users.append(
                User(
                    login=f"user{i}",
                    password_hash=hash_password("pass123"),
                    role=role,
                    is_active=True,
                    created_at=datetime.utcnow() - timedelta(days=i),
                )
            )
        db.add_all(users)
        db.flush()

        dict_codes = [
            ("category", "Категории"),
            ("style_type", "Типы изделий"),
            ("silhouette", "Силуэты"),
            ("fit_type", "Посадка"),
            ("shell_material", "Материалы верха"),
            ("lining_material", "Материалы подкладки"),
            ("insulation", "Утеплители"),
            ("sample_stage", "Этапы образца"),
            ("color", "Цвета"),
            ("size", "Размеры"),
        ]
        dictionaries = []
        for code, name in dict_codes:
            dictionaries.append(Dictionary(code=code, name=name, description=f"Справочник {name}"))
        db.add_all(dictionaries)
        db.flush()

        item_rows = []
        for d_idx, d in enumerate(dictionaries, start=1):
            for i in range(10):
                item_rows.append(
                    DictionaryItem(
                        dictionary_id=d.id,
                        code=f"{d.code}_{i+1}",
                        label=f"{d.name} {i+1}",
                        sort_order=i + 1,
                        is_active=True,
                    )
                )
        db.add_all(item_rows)
        db.flush()

        dict_by_code = {d.code: d for d in dictionaries}

        attributes = [
            Attribute(code="brand", name="Бренд", data_type="string", is_required=True, is_multivalue=False, is_active=True),
            Attribute(code="model_name", name="Название модели", data_type="string", is_required=True, is_multivalue=False, is_active=True),
            Attribute(code="drop_date", name="Дата дропа", data_type="date", is_required=False, is_multivalue=False, is_active=True),
            Attribute(code="retail_price", name="Розничная цена", data_type="number", is_required=True, is_multivalue=False, is_active=True),
            Attribute(code="is_waterproof", name="Водозащита", data_type="bool", is_required=False, is_multivalue=False, is_active=True),
            Attribute(code="main_color", name="Основной цвет", data_type="enum", is_required=True, is_multivalue=False, dictionary_id=dict_by_code["color"].id, is_active=True),
            Attribute(code="available_sizes", name="Размеры", data_type="enum", is_required=True, is_multivalue=True, dictionary_id=dict_by_code["size"].id, is_active=True),
            Attribute(code="style", name="Тип изделия", data_type="enum", is_required=True, is_multivalue=False, dictionary_id=dict_by_code["style_type"].id, is_active=True),
            Attribute(code="insulation_level", name="Утепление", data_type="enum", is_required=False, is_multivalue=False, dictionary_id=dict_by_code["insulation"].id, is_active=True),
            Attribute(code="capsule", name="Капсула", data_type="string", is_required=False, is_multivalue=False, is_active=True),
        ]
        db.add_all(attributes)
        db.flush()

        collections = []
        for i in range(10):
            collections.append(
                Collection(
                    code=f"FW{26 + i}",
                    name=f"Осень-Зима {2026 + i}",
                    season="FW" if i % 2 == 0 else "SS",
                    year=2026 + i,
                    brand_line="Women Outerwear",
                    is_active=True,
                )
            )
        db.add_all(collections)
        db.flush()

        suppliers = []
        countries = ["Turkey", "China", "Italy", "Vietnam", "Portugal", "India", "Bangladesh", "Romania", "Poland", "Serbia"]
        for i in range(10):
            suppliers.append(
                Supplier(
                    code=f"SUP-{i+1:03}",
                    name=f"Supplier {i+1}",
                    country=countries[i],
                    contact_email=f"factory{i+1}@example.com",
                    is_active=True,
                )
            )
        db.add_all(suppliers)
        db.flush()

        queues = []
        for i in range(10):
            queues.append(
                TaskQueue(
                    code=f"Q-{i+1:02}",
                    name=f"Очередь {i+1}",
                    description=f"Очередь задач {i+1}",
                    is_active=True,
                )
            )
        db.add_all(queues)
        db.flush()

        boards = []
        for i in range(10):
            boards.append(
                TaskBoard(
                    code=f"B-{i+1:02}",
                    name=f"Доска {i+1}",
                    description=f"Канбан-доска {i+1}",
                    is_active=True,
                )
            )
        db.add_all(boards)
        db.flush()

        category_items = db.query(DictionaryItem).filter(DictionaryItem.dictionary_id == dict_by_code["category"].id).order_by(DictionaryItem.sort_order.asc()).all()
        products = []
        for i in range(10):
            products.append(
                Product(
                    sku=f"WOC-{1000+i}",
                    name=f"Женское пальто модель {i+1}",
                    description=f"Демо-модель верхней одежды {i+1}",
                    status="active" if i % 3 else "draft",
                    category_item_id=category_items[i % len(category_items)].id,
                    created_at=datetime.utcnow() - timedelta(days=i * 2),
                    created_by=users[i % len(users)].id,
                    updated_at=datetime.utcnow() - timedelta(days=i),
                    updated_by=users[(i + 1) % len(users)].id,
                )
            )
        db.add_all(products)
        db.flush()

        specs = []
        for i, p in enumerate(products):
            specs.append(
                ProductSpec(
                    product_id=p.id,
                    collection_id=collections[i].id,
                    supplier_id=suppliers[i].id,
                    style_type="coat" if i % 2 == 0 else "puffer",
                    silhouette="oversize" if i % 2 == 0 else "regular",
                    fit_type="relaxed" if i % 3 == 0 else "regular",
                    length_cm=Decimal("95.0") + i,
                    shell_material="Wool blend" if i % 2 == 0 else "Polyamide",
                    lining_material="Viscose",
                    insulation="Down" if i % 2 else "Synthetic",
                    sample_stage=["proto", "salesman_sample", "pp_sample", "production"][i % 4],
                    planned_cost=Decimal("120.00") + Decimal(str(i * 8)),
                    actual_cost=Decimal("125.00") + Decimal(str(i * 8)),
                )
            )
        db.add_all(specs)
        db.flush()

        assignments = []
        values = []
        color_items = db.query(DictionaryItem).filter(DictionaryItem.dictionary_id == dict_by_code["color"].id).order_by(DictionaryItem.sort_order.asc()).all()
        for i in range(10):
            product = products[i]
            attr = attributes[i % len(attributes)]
            assignment = ProductAttributeAssignment(product_id=product.id, attribute_id=attr.id)
            assignments.append(assignment)
            db.add(assignment)
            db.flush()

            if attr.data_type == "string":
                values.append(ProductAttributeValue(assignment_id=assignment.id, value_string=f"Value {i+1}"))
            elif attr.data_type == "number":
                values.append(ProductAttributeValue(assignment_id=assignment.id, value_number=Decimal("99.90") + i))
            elif attr.data_type == "date":
                values.append(ProductAttributeValue(assignment_id=assignment.id, value_date=date(2026, 1, 1) + timedelta(days=i)))
            elif attr.data_type == "bool":
                values.append(ProductAttributeValue(assignment_id=assignment.id, value_bool=(i % 2 == 0)))
            elif attr.data_type == "enum":
                values.append(
                    ProductAttributeValue(
                        assignment_id=assignment.id,
                        dictionary_item_id=color_items[i % len(color_items)].id,
                    )
                )

        db.add_all(values)

        statuses = ["backlog", "todo", "in_progress", "review", "done"]
        priorities = ["low", "medium", "high", "critical"]
        tasks = []
        for i in range(10):
            tasks.append(
                Task(
                    title=f"Задача {i+1} по коллекции",
                    comment=f"Комментарий к задаче {i+1}",
                    status=statuses[i % len(statuses)],
                    priority=priorities[i % len(priorities)],
                    tags=f"fw{26+i},outerwear",
                    start_date=date(2026, 1, 1) + timedelta(days=i),
                    end_date=date(2026, 1, 10) + timedelta(days=i),
                    deadline=date(2026, 1, 15) + timedelta(days=i),
                    author_id=users[i % len(users)].id,
                    assignee_id=users[(i + 1) % len(users)].id,
                    queue_id=queues[i].id,
                    board_id=boards[i].id,
                    collection_id=collections[i].id,
                    product_id=products[i].id,
                )
            )
        db.add_all(tasks)
        db.commit()
        print("Seed complete: 10+ demo rows inserted per core table.")
    finally:
        db.close()


if __name__ == "__main__":
    reset_db()
    seed()
