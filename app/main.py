"""
Main FastAPI application.

Wired to Member 1's real ai/engine.py (FaceEngine class), which does its own
face detection AND matching internally (reads registered embeddings from
embeddings/*.npy on disk and returns registration numbers directly).
This backend's job is: receive the image, call the engine, translate its
result into attendance records, and persist them.
"""

import json
import logging
import datetime
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
import os
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db, engine, migrate_schema
from app import models
from app.schemas import AttendanceResponse, AttendanceRecordOut, StudentOut
from app.services.ai_service import (
    recognize_classroom_image,
    register_student_face,
    delete_student_embedding,
)
from app.services.excel_service import generate_attendance_excel
# for deletion of data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attendance_backend")

models.Base.metadata.create_all(bind=engine)
migrate_schema()

app = FastAPI(title="Smart Attendance Backend")
# Serve uploaded classroom images
UPLOAD_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "uploads"
)

os.makedirs(UPLOAD_ROOT, exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_ROOT),
    name="uploads"
)

# ----------------------------------------------------------------------
# Get all registered students
# ----------------------------------------------------------------------
@app.get("/students/", response_model=list[StudentOut])
def get_students(db: Session = Depends(get_db)):
    return db.query(models.Student).all()

# ----------------------------------------------------------------------
# Student registration
# ----------------------------------------------------------------------
@app.post("/students/")
def register_student(
    registration_number: str = Form(...),
    name: str = Form(...),
    course_name: str = Form(...),
    course_id: str = Form(...),
    class_id: str = Form(...),
    slot_no: str = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    import os
    import tempfile

    registration_number = registration_number.strip()
    name = name.strip()
    course_name = course_name.strip()
    course_id = course_id.strip()
    class_id = class_id.strip()
    slot_no = slot_no.strip()

    if not all([
        registration_number, name, course_name,
        course_id, class_id, slot_no
    ]):
        raise HTTPException(
            status_code=400,
            detail="All six registration fields are required."
        )

    if photo.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image."
        )

    existing = db.query(models.Student).filter(
        models.Student.registration_number == registration_number
    ).first()

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Student already registered."
        )

    image_bytes = photo.file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(image_bytes)

        result = register_student_face(registration_number, temp_path)

        student = models.Student(
            registration_number=registration_number,
            name=name,
            course_name=course_name,
            course_id=course_id,
            class_id=class_id,
            slot_no=slot_no,
            department=None,
            year=None,
            section=None,
            embedding=json.dumps(np.asarray(result).tolist()),
        )

        db.add(student)
        db.commit()
        db.refresh(student)

        logger.info("Registered student %s", registration_number)
        return StudentOut.model_validate(student)

    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(e))

    except SQLAlchemyError:
        db.rollback()
        logger.exception("Database error during student registration")
        raise HTTPException(
            status_code=500,
            detail="Database error during registration."
        )

    except Exception:
        db.rollback()
        logger.exception("Student registration failed")
        raise HTTPException(
            status_code=500,
            detail="Face registration failed."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------------------------------------------------------------
# Core workflow: image -> AI recognition -> attendance
# ----------------------------------------------------------------------

@app.post("/recognize-only/")
def recognize_only(image: UploadFile = File(...)):
    """Run face recognition without writing attendance records."""

    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image",
        )

    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    import os
    import tempfile

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        result = recognize_classroom_image(temp_path)

        return {
            "face_count": result["face_count"],
            "recognized_students": result["recognized_students"],
            "unknown_faces": result["unknown_faces"],
            "threshold": 0.50,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Recognition-only test failed")
        raise HTTPException(
            status_code=422,
            detail="Could not process image",
        ) from e

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/recognize-and-mark/", response_model=AttendanceResponse)
def recognize_and_mark(
    course_name: str = Form(...),
    course_id: str = Form(...),
    class_id: str = Form(...),
    slot_no: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    import os
    import uuid

    course_name = course_name.strip()
    course_id = course_id.strip().upper()
    class_id = class_id.strip().upper()
    slot_no = slot_no.strip()

    if not all([course_name, course_id, class_id, slot_no]):
        raise HTTPException(
            status_code=400,
            detail="Course Name, Course ID, Class ID and Slot Number are required."
        )

    if image.content_type not in (
        "image/jpeg",
        "image/png",
        "image/jpg"
    ):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image."
        )

    image_bytes = image.file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded."
        )

    # Fetch students belonging to the exact course, class and slot.
    all_students = db.query(models.Student).filter(
        models.Student.course_name == course_name,
        models.Student.course_id == course_id,
        models.Student.class_id == class_id,
        models.Student.slot_no == slot_no,
    ).all()

    if not all_students:
        raise HTTPException(
            status_code=404,
            detail="No students registered for this course, class and slot."
        )

    # Create permanent classroom image directory.
    upload_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "uploads",
        "classroom_images"
    )

    os.makedirs(upload_dir, exist_ok=True)

    extension = (
        ".png"
        if image.content_type == "image/png"
        else ".jpg"
    )

    filename = f"classroom_{uuid.uuid4().hex}{extension}"
    saved_image_path = os.path.join(upload_dir, filename)

    image_saved = False

    try:
        # Save the original classroom image.
        with open(saved_image_path, "wb") as image_file:
            image_file.write(image_bytes)

        image_saved = True

        # Keep a relative path in the database.
        image_path = os.path.relpath(
            saved_image_path,
            os.path.dirname(os.path.abspath(__file__))
        )

        eligible_numbers = [
            student.registration_number
            for student in all_students
        ]

        # Run face recognition only against selected-slot students.
        result = recognize_classroom_image(
            saved_image_path,
            registration_numbers=eligible_numbers
        )

        eligible_set = set(eligible_numbers)

        recognized_reg_numbers = {
            item["registration_number"]
            for item in result["recognized_students"]
        } & eligible_set

        confidence_by_reg = {
            item["registration_number"]: item["confidence"]
            for item in result["recognized_students"]
            if item["registration_number"] in eligible_set
        }

        unknown_count = len(result["unknown_faces"])

        # Create attendance session.
        session = models.ClassSession(
            class_name=class_id,
            section=None,
            course_name=course_name,
            course_id=course_id,
            class_id=class_id,
            slot_no=slot_no,
            image_path=image_path,
        )

        db.add(session)
        db.flush()

        records = []

        # Mark every registered student in this slot.
        for student in all_students:
            if student.registration_number in recognized_reg_numbers:
                record = models.AttendanceRecord(
                    student_id=student.id,
                    class_session_id=session.id,
                    status=models.AttendanceStatus.PRESENT,
                    similarity_score=confidence_by_reg.get(
                        student.registration_number
                    ),
                )
            else:
                record = models.AttendanceRecord(
                    student_id=student.id,
                    class_session_id=session.id,
                    status=models.AttendanceStatus.ABSENT,
                )

            db.add(record)
            records.append(record)

        db.commit()
        db.refresh(session)

        present_count = len(recognized_reg_numbers)
        absent_count = len(all_students) - present_count

        logger.info(
            "Session %s: Course %s, Class %s, Slot %s: %s present, %s absent",
            session.id,
            course_id,
            class_id,
            slot_no,
            present_count,
            absent_count,
        )

        return AttendanceResponse(
            class_session_id=session.id,
            session_date=session.session_date,
            course_name=session.course_name,
            course_id=session.course_id,
            class_id=session.class_id,
            slot_no=session.slot_no,
            image_path=session.image_path,

            present_count=present_count,
            absent_count=absent_count,
            unknown_faces_detected=unknown_count,
            records=[
                AttendanceRecordOut.model_validate(record)
                for record in records
            ],
        )

    except HTTPException:
        db.rollback()
        if image_saved and os.path.exists(saved_image_path):
            os.remove(saved_image_path)
        raise

    except Exception:
        db.rollback()

        if image_saved and os.path.exists(saved_image_path):
            os.remove(saved_image_path)

        logger.exception("Attendance marking failed")

        raise HTTPException(
            status_code=422,
            detail="Could not process image or save attendance."
        )

# ----------------------------------------------------------------------
# Attendance Sessions List
# ----------------------------------------------------------------------
@app.get("/attendance/sessions")
def get_attendance_sessions(db: Session = Depends(get_db)):
    sessions = (
        db.query(models.ClassSession)
        .order_by(models.ClassSession.session_date.desc())
        .all()
    )

    result = []

    for session in sessions:
        records = (
            db.query(models.AttendanceRecord)
            .filter(
                models.AttendanceRecord.class_session_id == session.id
            )
            .all()
        )

        present_count = sum(
            1
            for r in records
            if r.status == models.AttendanceStatus.PRESENT
        )

        absent_count = sum(
            1
            for r in records
            if r.status == models.AttendanceStatus.ABSENT
        )

        result.append(
                {
                   "id": session.id,
                    "class_name": session.class_name,
                   "section": session.section,
                   "course_name": session.course_name,
                   "course_id": session.course_id,
                   "class_id": session.class_id,
                   "slot_no": session.slot_no,
                   "image_path": session.image_path,
                   "date": session.session_date,
                   "total": len(records),
                   "present": present_count,
                    "absent": absent_count,
                    
                }
        )

    return result



# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------
@app.get("/attendance/history/{class_session_id}", response_model=AttendanceResponse)
def get_attendance_history(class_session_id: int, db: Session = Depends(get_db)):
    session = db.get(models.ClassSession, class_session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Class session not found")

    records = db.query(models.AttendanceRecord).filter(
        models.AttendanceRecord.class_session_id == class_session_id
    ).all()

    present_count = sum(1 for r in records if r.status == models.AttendanceStatus.PRESENT)
    absent_count = sum(1 for r in records if r.status == models.AttendanceStatus.ABSENT)

    return AttendanceResponse(
        class_session_id=session.id,
        session_date=session.session_date,
        course_name=session.course_name,
        course_id=session.course_id,
        class_id=session.class_id,
        slot_no=session.slot_no,
        image_path=session.image_path,
        present_count=present_count,
        absent_count=absent_count,
        unknown_faces_detected=0,  # not persisted per-session; add a column if needed later
        records=[AttendanceRecordOut.model_validate(r) for r in records],
    )


# ----------------------------------------------------------------------
# Excel export
# ----------------------------------------------------------------------
@app.get("/attendance/report/{class_session_id}")
def download_attendance_report(class_session_id: int, db: Session = Depends(get_db)):
    try:
        buffer = generate_attendance_excel(db, class_session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Excel generation failed: {e}")
        raise HTTPException(status_code=500, detail="Could not generate report")

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=attendance_{class_session_id}.xlsx"},
    )

# deletion of student data
@app.delete("/students/{registration_number}")
def delete_student(registration_number: str, db: Session = Depends(get_db)):
    student = db.query(models.Student).filter(
        models.Student.registration_number == registration_number
    ).first()

    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")

    try:
        # Remove attendance records first - required by the FK relationship,
        # otherwise the DELETE on students fails with an integrity error.
        db.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.student_id == student.id
        ).delete()

        db.delete(student)
        db.commit()

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"DB error while deleting student {registration_number}: {e}")
        raise HTTPException(status_code=500, detail="Database error while deleting student")

    # Also remove the .npy embedding file - this is the real source of truth
    # the recognition engine reads from, not just the DB row.
    embedding_deleted = delete_student_embedding(registration_number)
    if not embedding_deleted:
        logger.warning(f"No embedding file found on disk for {registration_number} (DB row still deleted)")

    logger.info(f"Deleted student {registration_number}")
    return {"message": f"Student {registration_number} deleted successfully"}