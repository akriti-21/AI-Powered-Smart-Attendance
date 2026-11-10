
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship, declarative_base

import enum
import datetime


Base = declarative_base()


# ---------------------------------------------------------
# Student Model
# ---------------------------------------------------------

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)

    registration_number = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    name = Column(String, nullable=False)

    # Existing fields retained for compatibility
    department = Column(String, nullable=True)
    year = Column(String, nullable=True)
    section = Column(String, nullable=True)

    # Course registration details
    course_name = Column(String, nullable=True)
    course_id = Column(String, nullable=True)
    class_id = Column(String, nullable=True)
    slot_no = Column(String, nullable=True)

    # Face embedding
    embedding = Column(String, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    attendance_records = relationship(
        "AttendanceRecord",
        back_populates="student"
    )


# ---------------------------------------------------------
# Class Attendance Session Model
# ---------------------------------------------------------

class ClassSession(Base):
    __tablename__ = "class_sessions"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)
    section = Column(String, nullable=True)

    # Course and class details
    course_name = Column(String, nullable=True)
    course_id = Column(String, nullable=True)
    class_id = Column(String, nullable=True)
    slot_no = Column(String, nullable=True)

    # Original classroom image location
    image_path = Column(String, nullable=True)

    session_date = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    attendance_records = relationship(
        "AttendanceRecord",
        back_populates="class_session"
    )


# ---------------------------------------------------------
# Attendance Status
# ---------------------------------------------------------

class AttendanceStatus(str, enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"


# ---------------------------------------------------------
# Attendance Record Model
# ---------------------------------------------------------

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "class_session_id",
            name="uq_student_session"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        nullable=False
    )

    class_session_id = Column(
        Integer,
        ForeignKey("class_sessions.id"),
        nullable=False
    )

    status = Column(
        SAEnum(AttendanceStatus),
        nullable=False
    )

    similarity_score = Column(Float, nullable=True)

    marked_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    student = relationship(
        "Student",
        back_populates="attendance_records"
    )

    class_session = relationship(
        "ClassSession",
        back_populates="attendance_records"
    )
