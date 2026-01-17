from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from db import Base

class NewsRow(Base):
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    headline: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)

