"""
SQLAlchemy models for the GLOBALISE document archive.

Entities: Series, Inventory, Document, Scan, Page, DocumentType, Settlement
and the junction / helper tables that connect them.
"""

import uuid
import enum
from typing import Optional, List
from datetime import date as Date

from sqlalchemy import (
    String,
    Integer,
    Boolean,
    Text,
    Date as DateType,
    ForeignKey,
    Enum as SQLEnum,
    Table,
    Column,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# Association table for Inventory-Series many-to-many relationship
inventory_series = Table(
    "inventory_series",
    Base.metadata,
    Column("inventory_id", String(36), ForeignKey("inventory.id"), primary_key=True),
    Column("series_id", String(36), ForeignKey("series.id"), primary_key=True),
)


class Series(Base):
    __tablename__ = "series"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(Text)
    part_of_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("series.id"), index=True
    )

    # Relationships
    part_of: Mapped[Optional["Series"]] = relationship(
        "Series",
        remote_side="Series.id",
        foreign_keys=[part_of_id],
        back_populates="sub_series",
    )
    sub_series: Mapped[List["Series"]] = relationship(
        "Series", foreign_keys="Series.part_of_id", back_populates="part_of"
    )
    inventories: Mapped[List["Inventory"]] = relationship(
        "Inventory", secondary=inventory_series, back_populates="member_of_series"
    )

    def __repr__(self):
        return f"<Series(title='{self.title[:50]}...')>"

    def __str__(self):
        return self.title


class Inventory(Base):
    __tablename__ = "inventory"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    inventory_number: Mapped[str] = mapped_column(String(10), index=True, unique=True)
    na_identifier: Mapped[Optional[str]] = mapped_column(String(36))
    handle: Mapped[Optional[str]] = mapped_column(String(255))
    date_start: Mapped[Optional[Date]] = mapped_column(DateType)
    date_end: Mapped[Optional[Date]] = mapped_column(DateType)

    # Relationships
    titles: Mapped[List["InventoryTitle"]] = relationship(
        "InventoryTitle", back_populates="inventory", cascade="all, delete-orphan"
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="inventory"
    )
    scans: Mapped[List["Scan"]] = relationship("Scan", back_populates="inventory")
    pages: Mapped[List["Page"]] = relationship("Page", back_populates="inventory")
    member_of_series: Mapped[List["Series"]] = relationship(
        "Series", secondary=inventory_series, back_populates="inventories"
    )

    def __repr__(self):
        return f"<Inventory(inventory_number='{self.inventory_number}')>"

    def __str__(self):
        return self.inventory_number


class InventoryTitle(Base):
    __tablename__ = "inventory_title"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(Text)
    inventory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("inventory.id"), index=True
    )

    # Relationships
    inventory: Mapped["Inventory"] = relationship("Inventory", back_populates="titles")

    def __repr__(self):
        return f"<InventoryTitle(title='{self.title[:50]}...')>"

    def __str__(self):
        return self.title


class DocumentIdentificationMethod(Base):
    __tablename__ = "document_identification_method"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    date: Mapped[Optional[Date]] = mapped_column(DateType)
    url: Mapped[Optional[str]] = mapped_column(String(255))

    # Relationships
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="method"
    )

    def __repr__(self):
        return f"<DocumentIdentificationMethod(name='{self.name}')>"

    def __str__(self):
        return f"{self.name} ({self.date})" if self.date else self.name


class Settlement(Base):
    """
    A canonical settlement entity drawn from location_index.csv.

    Each Settlement has a unique GLOBALISE identifier (glob_id, e.g. 'GLOB2_894')
    and one or more textual labels stored in SettlementLabel.  Multiple spelling
    variants and alternative names all resolve to the same Settlement row.
    """

    __tablename__ = "settlement"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    glob_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="GLOBALISE identifier, e.g. GLOB2_894",
    )

    # Relationships
    labels: Mapped[List["SettlementLabel"]] = relationship(
        "SettlementLabel", back_populates="settlement", cascade="all, delete-orphan"
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="location"
    )

    def __repr__(self) -> str:
        return f"<Settlement(glob_id='{self.glob_id}')>"

    def __str__(self) -> str:
        # Return the first label if available, otherwise the glob_id
        if self.labels:
            return self.labels[0].label
        return self.glob_id


class SettlementLabel(Base):
    """
    A textual label (name or spelling variant) for a Settlement.

    One Settlement may have several SettlementLabel rows — for example
    'Banjarmasin', 'Banjermassin', and 'Banjermassing' all belong to GLOB_523.
    The label column is indexed to support fast lookup during OBP import.
    """

    __tablename__ = "settlement_label"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    label: Mapped[str] = mapped_column(
        String(255), index=True, comment="Settlement name or spelling variant"
    )
    settlement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("settlement.id"), index=True
    )

    # Relationships
    settlement: Mapped["Settlement"] = relationship(
        "Settlement", back_populates="labels"
    )

    def __repr__(self) -> str:
        return f"<SettlementLabel(label='{self.label}', settlement_id='{self.settlement_id}')>"

    def __str__(self) -> str:
        return self.label


class Document(Base):
    __tablename__ = "document"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    inventory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("inventory.id"), index=True
    )
    title: Mapped[Optional[str]] = mapped_column(Text)
    date_earliest_begin: Mapped[Optional[Date]] = mapped_column(DateType, index=True)
    date_latest_begin: Mapped[Optional[Date]] = mapped_column(DateType)
    date_earliest_end: Mapped[Optional[Date]] = mapped_column(DateType)
    date_latest_end: Mapped[Optional[Date]] = mapped_column(DateType)
    date_text: Mapped[Optional[str]] = mapped_column(Text)
    part_of_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("document.id"), index=True
    )
    location_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("settlement.id"), index=True,
        comment="FK to Settlement; populated from location_index.csv via OBP import"
    )
    folio_start: Mapped[Optional[int]] = mapped_column(
        Integer, index=True,
        comment="First folio number of the document as recorded in the OBP index"
    )
    folio_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        comment="Last folio number of the document as recorded in the OBP index"
    )
    method_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_identification_method.id"), index=True
    )

    # Relationships
    inventory: Mapped["Inventory"] = relationship(
        "Inventory", back_populates="documents"
    )
    method: Mapped["DocumentIdentificationMethod"] = relationship(
        "DocumentIdentificationMethod", back_populates="documents"
    )
    part_of: Mapped[Optional["Document"]] = relationship(
        "Document",
        remote_side="Document.id",
        foreign_keys=[part_of_id],
        back_populates="sub_documents",
    )
    sub_documents: Mapped[List["Document"]] = relationship(
        "Document", foreign_keys="Document.part_of_id", back_populates="part_of"
    )
    location: Mapped[Optional["Settlement"]] = relationship(
        "Settlement", back_populates="documents"
    )
    document_types: Mapped[List["Document2Type"]] = relationship(
        "Document2Type", back_populates="document", cascade="all, delete-orphan"
    )
    document_types_linked: Mapped[List["Document2DocumentType"]] = relationship(
        "Document2DocumentType", back_populates="document", cascade="all, delete-orphan"
    )  # Links to DocumentType via Document2DocumentType
    external_ids: Mapped[List["Document2ExternalID"]] = relationship(
        "Document2ExternalID", back_populates="document", cascade="all, delete-orphan"
    )
    pages: Mapped[List["Page2Document"]] = relationship(
        "Page2Document", back_populates="document", cascade="all, delete-orphan"
    )

    @property
    def number_of_pages(self):
        """Get the number of pages in the document."""
        return len(self.pages)

    def __repr__(self):
        return f"<Document(title='{self.title}', inventory='{self.inventory_id}')>"

    def __str__(self):
        return (
            f"{self.title} ({self.inventory.inventory_number})"
            if self.title
            else f"Document {self.id}"
        )


class Document2Type(Base):
    __tablename__ = "document2type"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document.id"), index=True
    )
    document_type: Mapped[str] = mapped_column(String(255))

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="document_types"
    )

    def __repr__(self):
        return f"<Document2Type(type='{self.document_type}')>"


class ExternalID(Base):
    __tablename__ = "external_id"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    URL: Mapped[Optional[str]] = mapped_column(String(255))
    identifier: Mapped[Optional[str]] = mapped_column(String(255))
    context: Mapped[Optional[str]] = mapped_column(String(255))

    def __repr__(self):
        if self.URL:
            return f"<ExternalID(URL='{self.URL}')>"
        elif self.identifier:
            return f"<ExternalID(context='{self.context}', identifier='{self.identifier}')>"
        return f"<ExternalID(id='{self.id}')>"

    def __str__(self):
        if self.URL:
            return f"ExternalID[{self.URL}]"
        elif self.identifier:
            return f"ExternalID[{self.context}][{self.identifier}]"
        return f"ExternalID[{self.id}]"


class Document2ExternalID(Base):
    __tablename__ = "document2external_id"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document.id"), index=True
    )
    external_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("external_id.id"), index=True
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="external_ids"
    )
    external: Mapped["ExternalID"] = relationship("ExternalID")

    def __repr__(self):
        return f"<Document2ExternalID(document_id='{self.document_id}')>"


class DocumentType(Base):
    """A document-type concept drawn from the GLOBALISE/TANAP SKOS thesaurus."""

    __tablename__ = "document_type"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, comment="UUID extracted from the concept URI"
    )
    scheme: Mapped[str] = mapped_column(
        String(16), index=True, comment="GLOBALISE or TANAP"
    )
    pref_label_nl: Mapped[Optional[str]] = mapped_column(
        Text, comment="Dutch skos:prefLabel"
    )
    pref_label_en: Mapped[Optional[str]] = mapped_column(
        Text, comment="English skos:prefLabel"
    )

    # Relationships
    document_links: Mapped[List["Document2DocumentType"]] = relationship(
        "Document2DocumentType", back_populates="document_type"
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentType(id='{self.id}', scheme='{self.scheme}', "
            f"nl='{self.pref_label_nl}', en='{self.pref_label_en}')>"
        )

    def __str__(self) -> str:
        label = self.pref_label_en or self.pref_label_nl or self.id
        return f"{label} [{self.scheme}]"


class Document2DocumentType(Base):
    """
    Junction table linking a Document to a DocumentType concept from the thesaurus.

    Replaces the legacy free-text Document2Type for cases where a structured
    URI-based type is available (TANAP or GLOBALISE concept schemes).
    """

    __tablename__ = "document2documenttype"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document.id"), index=True
    )
    document_type_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_type.id"), index=True
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="document_types_linked"
    )
    document_type: Mapped["DocumentType"] = relationship(
        "DocumentType", back_populates="document_links"
    )

    def __repr__(self) -> str:
        return (
            f"<Document2DocumentType("
            f"document_id='{self.document_id}', "
            f"document_type_id='{self.document_type_id}')>"
        )


class PageType(str, enum.Enum):
    SINGLE = "Single"
    DOUBLE = "Double"
    OTHER = "Other"


class Scan(Base):
    __tablename__ = "scan"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(255), index=True)
    na_identifier: Mapped[Optional[str]] = mapped_column(String(36))
    iiif_image_info: Mapped[Optional[str]] = mapped_column(String(255))
    inventory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("inventory.id"), index=True
    )
    height: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    scan_type: Mapped[Optional[PageType]] = mapped_column(
        SQLEnum(PageType, values_callable=lambda obj: [e.value for e in obj])
    )

    # Relationships
    inventory: Mapped["Inventory"] = relationship("Inventory", back_populates="scans")
    pages: Mapped[List["Page"]] = relationship("Page", back_populates="scan")

    def get_thumbnail_url(self):
        """Get IIIF thumbnail URL."""
        url = self.iiif_image_info
        if url is None:
            return None
        return url.replace("info.json", "full/200,/0/default.jpg")

    def get_image_url(self, size="500,"):
        """Get IIIF image URL with specific size."""
        url = self.iiif_image_info
        if url is None:
            return None
        return url.replace("info.json", f"full/{size}/0/default.jpg")

    def __repr__(self):
        return f"<Scan(filename='{self.filename}')>"

    def __str__(self):
        return self.filename


class RectoVerso(str, enum.Enum):
    RECTO = "Recto"
    VERSO = "Verso"


class Page(Base):
    __tablename__ = "page"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    page_or_folio_number: Mapped[Optional[str]] = mapped_column(String(255))
    recto_verso: Mapped[Optional[RectoVerso]] = mapped_column(
        SQLEnum(RectoVerso, values_callable=lambda obj: [e.name for e in obj])
    )
    header: Mapped[Optional[str]] = mapped_column(Text)
    inventory_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("inventory.id"), index=True
    )
    scan_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("scan.id"), index=True
    )
    detected_languages: Mapped[Optional[str]] = mapped_column(Text)
    rotation: Mapped[int] = mapped_column(Integer, default=0)
    signatures: Mapped[Optional[str]] = mapped_column(Text)
    has_marginalia: Mapped[Optional[bool]] = mapped_column(Boolean)
    has_table: Mapped[Optional[bool]] = mapped_column(Boolean)
    has_illustration: Mapped[Optional[bool]] = mapped_column(Boolean)
    has_print: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_blank: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Relationships
    inventory: Mapped[Optional["Inventory"]] = relationship(
        "Inventory", back_populates="pages"
    )
    scan: Mapped[Optional["Scan"]] = relationship("Scan", back_populates="pages")
    documents: Mapped[List["Page2Document"]] = relationship(
        "Page2Document", back_populates="page", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Page(scan='{self.scan_id}', recto_verso='{self.recto_verso}')>"

    def __str__(self):
        if self.scan and self.recto_verso:
            return f"Page for scan {self.scan.filename} ({self.recto_verso.value})"
        elif self.scan:
            return f"Page for scan {self.scan.filename}"
        return f"Page {self.id}"


class Page2Document(Base):
    __tablename__ = "page2document"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    page_id: Mapped[str] = mapped_column(String(36), ForeignKey("page.id"), index=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document.id"), index=True
    )
    index: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(
        String(32),
        default="BASELINE",
        server_default="BASELINE",
        comment="How this page-document link was established: BASELINE or FOLIO_RANGE",
    )
    confidence: Mapped[float] = mapped_column(
        default=1.0,
        server_default="1.0",
        comment="Confidence of the link: 1.0 = definitive (BASELINE), 0.0-1.0 = indicative",
    )

    # Relationships
    page: Mapped["Page"] = relationship("Page", back_populates="documents")
    document: Mapped["Document"] = relationship("Document", back_populates="pages")

    def __repr__(self):
        return (
            f"<Page2Document(page_id='{self.page_id}', document_id='{self.document_id}', "
            f"index={self.index}, source='{self.source}', confidence={self.confidence})>"
        )
