# AI-Powered Smart Attendance Management System

An AI-powered student attendance management system designed to automate attendance recording using facial recognition.

The project aims to identify registered students from classroom images, associate recognized faces with student registration numbers, and maintain attendance records digitally.

## Project Overview

Traditional attendance systems require instructors to mark attendance manually. This project explores an automated approach using facial recognition and a mobile application.

The system is being developed with an offline-first approach, with the goal of performing face detection, recognition, student registration, and attendance management locally on the device.

## Key Objectives

- Automate student identification using facial recognition.
- Register students using their details and photographs.
- Detect and recognize multiple faces in classroom images.
- Associate recognized faces with student registration numbers.
- Record and manage attendance.
- Maintain attendance history and generate reports.
- Minimize dependency on internet connectivity.

## Technology Stack

| Component | Technology |
|---|---|
| Mobile application | React Native |
| Framework | Expo |
| Android native integration | Kotlin |
| Programming languages | JavaScript, Kotlin, Python |
| Face detection | YOLOv11 Face, RetinaFace |
| Face recognition | ArcFace |
| Model format | ONNX |
| Inference runtime | ONNX Runtime |
| Image processing | OpenCV |
| Local database | SQLite |
| Version control | Git and Git LFS |

## Project Structure

```text
Smart-Attendance/
│
├── ai/                         # Python AI implementation
├── app/                        # Application-related source
├── docs/                       # Project documentation
├── experiments/                # AI experiments
├── tests/                      # Test files
│
├── mobile-app/
│   ├── screens/                # Register, Attendance, History
│   ├── android/
│   │   └── app/src/main/
│   │       ├── assets/models/  # ONNX model files
│   │       └── java/           # Native Kotlin integration
│   ├── App.js
│   └── database.js
│
├── requirements.txt
├── README.md
└── README_TESTING.md
```

## AI Models

The Android application includes the following ONNX models:

| Model | Purpose |
|---|---|
| YOLOv11 Face | Face detection |
| RetinaFace | Face detection and facial landmarks |
| ArcFace (w600k_r50) | 512-dimensional face embeddings |

Model files are stored under:

```text
mobile-app/android/app/src/main/assets/models/
```

The `.onnx` files are tracked using Git LFS because the ArcFace model exceeds GitHub's regular 100 MB file limit.

## Application Features

### Student Registration
- Enter student registration details.
- Capture a photograph using the camera or select one from the gallery.
- Store student information locally.
- View and delete registered students.

### Face Recognition
- Native Android integration using Kotlin.
- ONNX Runtime-based model loading.
- OpenCV-based image processing.
- Face embedding extraction using ArcFace.

### Attendance Management
- Attendance screen and history interface are present.
- Local attendance workflow and complete offline integration are still under development and testing.

## Current Development Status

| Module | Status |
|---|---|
| React Native application setup | Implemented |
| Registration screen UI | Implemented |
| Local student registration | Implemented |
| Attendance and history screens | UI implemented |
| Kotlin native module | Implemented |
| ONNX model integration | Implemented at code/build level |
| Android compilation | Successful |
| On-device model inference | Requires verification |
| Complete offline attendance workflow | In progress |
| Attendance report generation | In progress |

Features marked as in progress should not be considered production-ready.

## Getting Started

### Prerequisites

- macOS, Windows, or Linux
- Node.js and npm
- Java Development Kit compatible with the Android Gradle project
- Android Studio and Android SDK
- Python 3
- Git
- Git LFS

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Smart-Attendance
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

### 4. Start the development server

```bash
npx expo start --dev-client
```

For a USB-connected Android device, configure ADB reverse port forwarding:

```bash
adb reverse tcp:8081 tcp:8081
```

Open the development build on the connected device and connect to the local development server.

### 5. Build the Android application

From the Android project directory:

```bash
cd android
./gradlew assembleDebug
```

The debug APK is generated at:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

## Python Environment

For Python-based AI components, create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Collaboration Guidelines

This project uses Git for version control and GitHub for collaboration.

1. Clone the repository.
2. Create a separate branch for your task.
3. Make and test your changes.
4. Commit and push your branch.
5. Open a Pull Request into `main`.
6. Review and merge changes after approval.

Example:

```bash
git checkout -b feature/your-feature
git add .
git commit -m "Describe your changes"
git push -u origin feature/your-feature
```

Avoid committing:
- Virtual environments and dependency folders.
- Local databases and backups.
- Student photographs and biometric embeddings.
- API keys, passwords, and environment files.
- Generated APKs and build outputs.

## Privacy and Data Protection

Student photographs, facial embeddings, and attendance records are sensitive data.

Do not upload real student biometric data, personal photographs, or private attendance databases to the public repository. Use authorized test data and follow institutional privacy requirements.

## Project Status

This project is under active development as an academic software project. Its components are being integrated and tested.

## License

To be decided.
