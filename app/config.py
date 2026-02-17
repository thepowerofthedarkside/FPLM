from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "pim.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"
SECRET_KEY = "change-me-in-production"
