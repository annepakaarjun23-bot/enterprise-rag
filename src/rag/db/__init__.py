"""Create the registry tables. Safe to re-run (skips existing tables).

Usage: python scripts/init_db.py
"""

from rag.db.models import Base
from rag.db.session import engine

if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Tables ready:", ", ".join(Base.metadata.tables))