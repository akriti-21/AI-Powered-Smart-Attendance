
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class StudentOut(BaseModel):
    id: int
    registration_number: str
    name: str

    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None

    course_name: Optional[str] = None
    course_id: Optional[str] = None
    class_id: Optional[str] = None
    slot_no: Optional[str] = None

    model_config = {"from_attributes": True}


class AttendanceRecordOut(BaseModel):
    student: StudentOut
    status: str
    similarity_score: Optional[float] = None

    model_config = {"from_attributes": True}

class AttendanceResponse(BaseModel):
    class_session_id: int
    session_date: datetime

    course_name: Optional[str] = None
    course_id: Optional[str] = None
    class_id: Optional[str] = None
    slot_no: Optional[str] = None
    image_path: Optional[str] = None

    present_count: int
    absent_count: int
    unknown_faces_detected: int
    records: List[AttendanceRecordOut]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
