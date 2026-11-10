# AI-Powered Smart Attendance Management System

An offline-first student attendance management application that uses AI-powered facial recognition to identify students, automate attendance recording, and generate attendance reports.

The system combines a React Native mobile application with native Android AI integration to perform face detection, face recognition, student registration, and attendance management using locally stored data.

**Repository:** [AI-Powered-Smart-Attendance](https://github.com/akriti-21/AI-Powered-Smart-Attendance)

---

## Project Overview

Traditional classroom attendance is often recorded manually, consuming valuable teaching time and making attendance management difficult.

This project aims to automate the process through facial recognition. Instructors can capture a classroom image, detect multiple faces, identify registered students, and automatically record attendance.

The application follows an offline-first architecture, using locally stored face embeddings, ONNX models, and SQLite database storage to support core attendance operations without requiring a continuous internet connection.

## Key Features

### 1. Student Registration
- Register students using their name and registration number.
- Capture photographs using the camera or select images from the gallery.
- Generate facial embeddings using ArcFace.
- Store student information and embeddings locally.
- Retrieve registered student information from SQLite.

### 2. AI-Based Face Detection
- Detect multiple faces in classroom images.
- YOLOv11 Face for initial face detection.
- RetinaFace for face detection and facial landmarks.
- OpenCV for image processing.
- Additional full-image RetinaFace detection to improve face coverage.

### 3. Face Recognition
- ArcFace-based facial feature extraction.
- Generate 512-dimensional face embeddings.
- Compare detected face embeddings against registered student embeddings.
- Identify registered students using similarity scores.
- Classify unmatched detected faces as UNKNOWN.

### 4. Automated Attendance
The application classifies detected and registered students into:

| Status | Description |
|---|---|
| Present | Registered student successfully recognized |
| Absent | Registered student not detected in the attendance image |
| Unknown | Face detected but not matched with a registered student |

### 5. Attendance History
- Store attendance sessions locally.
- View previous attendance records.
- Display present and absent counts.
- Retrieve individual session details.

### 6. Attendance Report Generation
- Generate attendance reports locally in CSV format.
- Include registration numbers, student names, attendance status, and similarity scores.
- Open reports using compatible spreadsheet applications.

### 7. Offline-First Operation
- Local SQLite database.
- Locally stored student embeddings.
- On-device ONNX model inference.
- Native Android face recognition.
- No continuous backend connection required for the core local attendance workflow.

---

## Technology Stack

| Component | Technology |
|---|---|
| Mobile application | React Native |
| Framework | Expo |
| Android native integration | Kotlin |
| AI implementation | Python |
| Face detection | YOLOv11 Face, RetinaFace |
| Face recognition | ArcFace |
| Model format | ONNX |
| Inference runtime | ONNX Runtime |
| Image processing | OpenCV |
| Local database | SQLite |
| Version control | Git and Git LFS |

---

## System Architecture

```text
                 CLASSROOM IMAGE
                        |
                        v
                IMAGE PROCESSING
                    OpenCV
                        |
                        v
                 FACE DETECTION
                YOLOv11 + RetinaFace
                        |
                        v
                FACE ALIGNMENT
                        |
                        v
              FEATURE EXTRACTION
                     ArcFace
                        |
                        v
             512-D FACE EMBEDDINGS
                        |
                        v
              FACE MATCHING ENGINE
                        |
             +----------+----------+
             |          |          |
             v          v          v
          PRESENT     ABSENT     UNKNOWN
             |          |          |
             +----------+----------+
                        |
                        v
              SQLITE DATABASE
                        |
             +----------+----------+
             |                     |
             v                     v
       ATTENDANCE HISTORY     CSV REPORT
```

## AI Models

| Model | Purpose |
|---|---|
| YOLOv11 Face | Initial face detection |
| RetinaFace | Face detection and facial landmarks |
| ArcFace (w600k_r50) | 512-dimensional facial embeddings |

Model files are stored under:

```text
mobile-app/android/app/src/main/assets/models/
```

Git LFS is used for large model files.

---

## Project Structure

```text
AI-Powered-Smart-Attendance/
│
├── ai/                         # Python AI implementation
│
├── app/                        # Application-related source
│
├── docs/                       # Project documentation
│
├── experiments/
│   └── model_comparison/       # Model experiments
│
├── mobile-app/
│   ├── screens/                # Application screens
│   ├── android/
│   │   └── app/src/main/
│   │       ├── assets/models/  # ONNX models
│   │       └── java/           # Native Kotlin code
│   ├── App.js
│   └── database.js
│
├── tests/                      # Test files
│
├── requirements.txt
├── README.md
└── README_TESTING.md
```

---

## Getting Started

### Prerequisites

- macOS, Windows, or Linux
- Node.js and npm
- Java Development Kit 17
- Android Studio and Android SDK
- Python 3
- Git
- Git LFS

### 1. Clone the repository

```bash
git clone https://github.com/akriti-21/AI-Powered-Smart-Attendance.git

cd AI-Powered-Smart-Attendance
```

### 2. Download Git LFS files

```bash
git lfs install
git lfs pull
```

### 3. Install mobile application dependencies

```bash
cd mobile-app

npm install
```

### 4. Start the Expo development server

```bash
npx expo start --dev-client --localhost -c
```

For a USB-connected Android device:

```bash
adb reverse tcp:8081 tcp:8081
```

Open the installed development build on the Android device.

### 5. Compile the Android application

```bash
cd android

./gradlew :app:assembleDebug
```

The debug APK is generated at:

```text
app/build/outputs/apk/debug/app-debug.apk
```

### 6. Set up the Python environment

For Python-based AI components:

```bash
cd ../..

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Sample Testing Results

The following results were observed during a classroom-image attendance test:

| Metric | Result |
|---|---:|
| Registered students present | 2 |
| Registered students absent | 1 |
| Unknown faces detected | 14 |
| Total detected faces represented | 16 |
| Missing embeddings | 0 |

The application also successfully generated and opened a local CSV attendance report containing registration numbers, student names, statuses, and similarity scores.

**Note:** These are results from a particular test image, not a general face-recognition accuracy benchmark. Performance can vary with image quality, lighting, face size, occlusion, and camera position.

---

## Privacy and Data Protection

Facial photographs and biometric embeddings are sensitive information.

- Do not commit real student photographs or biometric embeddings to a public repository.
- Do not upload private attendance databases.
- Keep API credentials, signing keys, and passwords out of version control.
- Use authorized test data.
- Obtain appropriate consent before processing student biometric information.
- Follow applicable institutional privacy and data-protection requirements.

---

## Current Development Status

| Module | Status |
|---|---|
| React Native mobile application | Implemented |
| Student registration | Implemented and tested |
| SQLite local storage | Implemented and tested |
| YOLOv11 Face integration | Implemented |
| RetinaFace integration | Implemented |
| ArcFace embedding extraction | Implemented and tested |
| Registered student recognition | Tested |
| Unknown face detection | Tested |
| Attendance recording | Tested |
| Attendance history | Tested |
| Local CSV report generation | Tested |
| Complete offline operation | Requires final airplane-mode verification |
| Production deployment | Planned |

---

## Future Enhancements

- Improve face detection for small and partially occluded faces.
- Improve recognition accuracy under different lighting conditions.
- Add attendance analytics and visual dashboards.
- Support more export formats, including XLSX.
- Improve biometric data protection.
- Conduct larger-scale classroom testing.
- Prepare a production Android release.

---

## Collaboration

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Commit your changes.
5. Submit a pull request.

```bash
git checkout -b feature/your-feature

git add .

git commit -m "Add your feature"

git push -u origin feature/your-feature
```

---

## Project Information

**Project:** AI-Powered Smart Attendance Management System

**Repository:** [akriti-21/AI-Powered-Smart-Attendance](https://github.com/akriti-21/AI-Powered-Smart-Attendance)

**Primary focus:** Artificial Intelligence, Computer Vision, Mobile Application Development, and Offline Computing.

**License:** To be decided.

---

*Developed as an academic project exploring AI-powered facial recognition and offline student attendance automation.*
