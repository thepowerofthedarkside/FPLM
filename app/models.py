from datetime import datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    login: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(300), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    department: Mapped[str | None] = mapped_column(String(200), nullable=True)
    position: Mapped[str | None] = mapped_column(String(200), nullable=True)
    department_item_id: Mapped[int | None] = mapped_column(ForeignKey("dictionary_items.id"), nullable=True)
    position_item_id: Mapped[int | None] = mapped_column(ForeignKey("dictionary_items.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(30), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    department_item: Mapped["DictionaryItem | None"] = relationship("DictionaryItem", foreign_keys=[department_item_id])
    position_item: Mapped["DictionaryItem | None"] = relationship("DictionaryItem", foreign_keys=[position_item_id])


class Dictionary(Base):
    __tablename__ = "dictionaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    items: Mapped[list["DictionaryItem"]] = relationship(
        "DictionaryItem", back_populates="dictionary", cascade="all, delete-orphan"
    )


class DictionaryItem(Base):
    __tablename__ = "dictionary_items"
    __table_args__ = (UniqueConstraint("dictionary_id", "code", name="uq_dictionary_item_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dictionary_id: Mapped[int] = mapped_column(ForeignKey("dictionaries.id"), nullable=False)
    code: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    dictionary: Mapped[Dictionary] = relationship("Dictionary", back_populates="items")


class Attribute(Base):
    __tablename__ = "attributes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    group_code: Mapped[str] = mapped_column(String(40), default="fashion_spec", nullable=False)
    data_type: Mapped[str] = mapped_column(String(20), nullable=False)
    is_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_multivalue: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    dictionary_id: Mapped[int | None] = mapped_column(ForeignKey("dictionaries.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    dictionary: Mapped[Dictionary | None] = relationship("Dictionary")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    cover_image_path: Mapped[str | None] = mapped_column(String(300), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="draft", nullable=False)
    category_item_id: Mapped[int | None] = mapped_column(ForeignKey("dictionary_items.id"), nullable=True)
    designer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    product_manager_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    pattern_maker_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    technologist_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    department_head_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    updated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    category_item: Mapped[DictionaryItem | None] = relationship("DictionaryItem")
    designer: Mapped[User | None] = relationship("User", foreign_keys=[designer_id])
    product_manager: Mapped[User | None] = relationship("User", foreign_keys=[product_manager_id])
    pattern_maker: Mapped[User | None] = relationship("User", foreign_keys=[pattern_maker_id])
    technologist: Mapped[User | None] = relationship("User", foreign_keys=[technologist_id])
    department_head: Mapped[User | None] = relationship("User", foreign_keys=[department_head_id])
    spec: Mapped["ProductSpec | None"] = relationship(
        "ProductSpec", back_populates="product", cascade="all, delete-orphan", uselist=False
    )
    files: Mapped[list["ProductFile"]] = relationship(
        "ProductFile", back_populates="product", cascade="all, delete-orphan"
    )
    materials: Mapped[list["ProductMaterial"]] = relationship(
        "ProductMaterial", back_populates="product", cascade="all, delete-orphan"
    )
    bom_items: Mapped[list["ProductBOMItem"]] = relationship(
        "ProductBOMItem", back_populates="product", cascade="all, delete-orphan"
    )
    variants: Mapped[list["ProductVariant"]] = relationship(
        "ProductVariant", back_populates="product", cascade="all, delete-orphan"
    )
    samples: Mapped[list["ProductSample"]] = relationship(
        "ProductSample", back_populates="product", cascade="all, delete-orphan"
    )
    costing_items: Mapped[list["ProductCostingItem"]] = relationship(
        "ProductCostingItem", back_populates="product", cascade="all, delete-orphan"
    )


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    season: Mapped[str] = mapped_column(String(20), nullable=False)  # FW/SS
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    brand_line: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Supplier(Base):
    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str] = mapped_column(String(120), nullable=False)
    contact_email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    materials: Mapped[list["ProductMaterial"]] = relationship("ProductMaterial", back_populates="supplier")


class ProductSpec(Base):
    __tablename__ = "product_specs"
    __table_args__ = (UniqueConstraint("product_id", name="uq_product_spec_product"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    collection_id: Mapped[int | None] = mapped_column(ForeignKey("collections.id"), nullable=True)
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id"), nullable=True)
    style_type: Mapped[str | None] = mapped_column(String(80), nullable=True)  # coat/trench/puffer etc
    capsule: Mapped[str | None] = mapped_column(String(120), nullable=True)
    silhouette: Mapped[str | None] = mapped_column(String(80), nullable=True)
    fit_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    length_cm: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    shell_material: Mapped[str | None] = mapped_column(String(180), nullable=True)
    lining_material: Mapped[str | None] = mapped_column(String(180), nullable=True)
    insulation: Mapped[str | None] = mapped_column(String(180), nullable=True)
    sample_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)  # proto/salesman/pp
    planned_cost: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    actual_cost: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)

    product: Mapped[Product] = relationship("Product", back_populates="spec")
    collection: Mapped[Collection | None] = relationship("Collection")
    supplier: Mapped[Supplier | None] = relationship("Supplier")


class ProductFile(Base):
    __tablename__ = "product_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(350), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    product: Mapped[Product] = relationship("Product", back_populates="files")
    uploader: Mapped[User | None] = relationship("User")


class TaskQueue(Base):
    __tablename__ = "task_queues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class TaskBoard(Base):
    __tablename__ = "task_boards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    filter_queue_id: Mapped[int | None] = mapped_column(ForeignKey("task_queues.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    filter_queue: Mapped["TaskQueue | None"] = relationship("TaskQueue")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(220), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="todo")
    priority: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    tags: Mapped[str | None] = mapped_column(String(300), nullable=True)
    start_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    deadline: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    assignee_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    queue_id: Mapped[int | None] = mapped_column(ForeignKey("task_queues.id"), nullable=True)
    board_id: Mapped[int | None] = mapped_column(ForeignKey("task_boards.id"), nullable=True)
    collection_id: Mapped[int | None] = mapped_column(ForeignKey("collections.id"), nullable=True)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    author: Mapped[User | None] = relationship("User", foreign_keys=[author_id])
    assignee: Mapped[User | None] = relationship("User", foreign_keys=[assignee_id])
    queue: Mapped[TaskQueue | None] = relationship("TaskQueue")
    board: Mapped[TaskBoard | None] = relationship("TaskBoard")
    collection: Mapped[Collection | None] = relationship("Collection")
    product: Mapped[Product | None] = relationship("Product")
    files: Mapped[list["TaskFile"]] = relationship("TaskFile", back_populates="task", cascade="all, delete-orphan")


class TaskFile(Base):
    __tablename__ = "task_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(350), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    task: Mapped[Task] = relationship("Task", back_populates="files")
    uploader: Mapped[User | None] = relationship("User")


class SystemSetting(Base):
    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entity_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scope_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    scope_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    actor_login: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    actor: Mapped[User | None] = relationship("User")


class ProductAttributeAssignment(Base):
    __tablename__ = "product_attributes"
    __table_args__ = (UniqueConstraint("product_id", "attribute_id", name="uq_product_attribute"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    attribute_id: Mapped[int] = mapped_column(ForeignKey("attributes.id"), nullable=False)

    product: Mapped[Product] = relationship("Product")
    attribute: Mapped[Attribute] = relationship("Attribute")
    values: Mapped[list["ProductAttributeValue"]] = relationship(
        "ProductAttributeValue", back_populates="assignment", cascade="all, delete-orphan"
    )


class ProductAttributeValue(Base):
    __tablename__ = "product_attribute_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("product_attributes.id"), nullable=False)
    value_string: Mapped[str | None] = mapped_column(Text, nullable=True)
    value_number: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    value_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    value_bool: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    dictionary_item_id: Mapped[int | None] = mapped_column(ForeignKey("dictionary_items.id"), nullable=True)

    assignment: Mapped[ProductAttributeAssignment] = relationship("ProductAttributeAssignment", back_populates="values")
    dictionary_item: Mapped[DictionaryItem | None] = relationship("DictionaryItem")


class UserNotificationPreference(Base):
    __tablename__ = "user_notification_preferences"
    __table_args__ = (UniqueConstraint("user_id", "event_key", name="uq_user_notification_preference"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    event_key: Mapped[str] = mapped_column(String(80), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    user: Mapped[User] = relationship("User")


class UserNotification(Base):
    __tablename__ = "user_notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    event_key: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    link_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    entity_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship("User")


class EntityComment(Base):
    __tablename__ = "entity_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    author: Mapped[User | None] = relationship("User")
    revisions: Mapped[list["EntityCommentRevision"]] = relationship(
        "EntityCommentRevision", back_populates="comment", cascade="all, delete-orphan"
    )


class EntityCommentRevision(Base):
    __tablename__ = "entity_comment_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    comment_id: Mapped[int] = mapped_column(ForeignKey("entity_comments.id"), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(String(30), nullable=False)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    comment: Mapped[EntityComment] = relationship("EntityComment", back_populates="revisions")
    actor: Mapped[User | None] = relationship("User")


class ProductMaterial(Base):
    __tablename__ = "product_materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id"), nullable=True)
    code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    composition: Mapped[str | None] = mapped_column(String(255), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    price_per_unit: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), default="RUB", nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    product: Mapped[Product] = relationship("Product", back_populates="materials")
    supplier: Mapped[Supplier | None] = relationship("Supplier", back_populates="materials")


class ProductBOMItem(Base):
    __tablename__ = "product_bom_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    material_id: Mapped[int | None] = mapped_column(ForeignKey("product_materials.id"), nullable=True)
    component: Mapped[str] = mapped_column(String(200), nullable=False)
    quantity: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    waste_percent: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    product: Mapped[Product] = relationship("Product", back_populates="bom_items")
    material: Mapped[ProductMaterial | None] = relationship("ProductMaterial")


class ProductVariant(Base):
    __tablename__ = "product_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    color: Mapped[str] = mapped_column(String(100), nullable=False)
    size: Mapped[str] = mapped_column(String(40), nullable=False)
    sku_suffix: Mapped[str | None] = mapped_column(String(60), nullable=True)
    ean: Mapped[str | None] = mapped_column(String(50), nullable=True)
    planned_qty: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str | None] = mapped_column(String(40), nullable=True)

    product: Mapped[Product] = relationship("Product", back_populates="variants")


class ProductSample(Base):
    __tablename__ = "product_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    sample_type: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    due_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    received_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    product: Mapped[Product] = relationship("Product", back_populates="samples")
    owner: Mapped[User | None] = relationship("User")


class ProductCostingItem(Base):
    __tablename__ = "product_costing_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    cost_group: Mapped[str] = mapped_column(String(80), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="RUB", nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    product: Mapped[Product] = relationship("Product", back_populates="costing_items")
