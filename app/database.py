
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./attendance.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def migrate_schema():
    migrations = {
        "students": {
            "course_name": "TEXT",
            "course_id": "TEXT",
            "class_id": "TEXT",
            "slot_no": "TEXT",
        },
        "class_sessions": {
            "course_name": "TEXT",
            "course_id": "TEXT",
            "class_id": "TEXT",
            "slot_no": "TEXT",
            "image_path": "TEXT",
        },
    }

    with engine.begin() as connection:
        for table_name, columns in migrations.items():
            existing_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            }

            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    connection.exec_driver_sql(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN {column_name} {column_type}"
                    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
