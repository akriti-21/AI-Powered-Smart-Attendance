import React, { useState } from 'react';

import {
  View,
  Text,
  TouchableOpacity,
  Alert,
  StyleSheet,
  Image,
  ActivityIndicator,
  TextInput,
  ScrollView,
  NativeModules,
} from 'react-native';

import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { getDatabase } from '../database';

const SIMILARITY_THRESHOLD = 0.45;

// ======================================================
// EMBEDDING UTILITIES
// ======================================================

function parseEmbedding(value) {
  try {
    const parsed =
      typeof value === 'string' ? JSON.parse(value) : value;

    if (!Array.isArray(parsed) || parsed.length !== 512) {
      return null;
    }

    const vector = parsed.map(Number);

    if (!vector.every(Number.isFinite)) {
      return null;
    }

    return vector;
  } catch {
    return null;
  }
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) {
    return -1;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (!normA || !normB) {
    return -1;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ======================================================
// ATTENDANCE SCREEN
// ======================================================

export default function AttendanceScreen() {
  const [photo, setPhoto] = useState(null);

  const [courseName, setCourseName] = useState('');
  const [courseId, setCourseId] = useState('');
  const [classId, setClassId] = useState('');
  const [slotNo, setSlotNo] = useState('');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // ====================================================
  // TAKE CLASSROOM PHOTO
  // ====================================================

  const takePhoto = async () => {
    try {
      const permission =
        await ImagePicker.requestCameraPermissionsAsync();

      if (!permission.granted) {
        Alert.alert(
          'Permission required',
          'Camera permission is required to take a classroom photo.'
        );
        return;
      }

      const pickerResult =
        await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: false,
          quality: 0.8,
        });

      if (
        !pickerResult.canceled &&
        pickerResult.assets?.length > 0
      ) {
        setPhoto(pickerResult.assets[0]);
        setResult(null);
      }
    } catch (error) {
      Alert.alert(
        'Camera error',
        error?.message || 'Unable to open camera.'
      );
    }
  };

  // ====================================================
  // CHOOSE CLASSROOM PHOTO
  // ====================================================

  const choosePhoto = async () => {
    try {
      const permission =
        await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (!permission.granted) {
        Alert.alert(
          'Permission required',
          'Gallery permission is required to choose a classroom photo.'
        );
        return;
      }

      const pickerResult =
        await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: false,
          quality: 0.8,
        });

      if (
        !pickerResult.canceled &&
        pickerResult.assets?.length > 0
      ) {
        setPhoto(pickerResult.assets[0]);
        setResult(null);
      }
    } catch (error) {
      Alert.alert(
        'Gallery error',
        error?.message || 'Unable to choose a photo.'
      );
    }
  };

  // ====================================================
  // MARK ATTENDANCE OFFLINE
  // ====================================================

  const markAttendance = async () => {
    const selectedCourseName = courseName.trim();
    const selectedCourseId = courseId.trim().toUpperCase();
    const selectedClassId = classId.trim().toUpperCase();
    const selectedSlot = slotNo.trim().toUpperCase();

    if (
      !selectedCourseName ||
      !selectedCourseId ||
      !selectedClassId ||
      !selectedSlot
    ) {
      Alert.alert(
        'Missing information',
        'Enter Course Name, Course ID, Class ID and Slot Number.'
      );
      return;
    }

    if (!photo?.uri) {
      Alert.alert(
        'Photo required',
        'Choose a classroom photo first.'
      );
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      // ----------------------------------------------
      // 1. CONNECT TO LOCAL SQLITE DATABASE
      // ----------------------------------------------

      const db = await getDatabase();

      // ----------------------------------------------
      // 2. FETCH REGISTERED STUDENTS
      // ----------------------------------------------

      const students = await db.getAllAsync(
        `SELECT
           registration_number,
           name,
           embedding,
           photo_uri
         FROM students
         WHERE TRIM(UPPER(COALESCE(course_id, ''))) = ?
           AND TRIM(UPPER(COALESCE(class_id, ''))) = ?
           AND TRIM(
             UPPER(COALESCE(CAST(slot_no AS TEXT), ''))
           ) = ?`,
        [
          selectedCourseId,
          selectedClassId,
          selectedSlot,
        ]
      );

      console.log(
        'REGISTERED STUDENTS:',
        students.length
      );

      if (!students.length) {
        throw new Error(
          'No students found for this course, class and slot. Check the entered details.'
        );
      }

      console.log(
        'STUDENT DETAILS:',
        students.map(student => ({
          registrationNumber: student.registration_number,
          name: student.name,
          hasEmbedding: !!parseEmbedding(student.embedding),
          photo: student.photo_uri,
        }))
      );

      // ----------------------------------------------
      // 3. INITIALIZE OFFLINE FACE RECOGNITION
      // ----------------------------------------------

      const faceInference = NativeModules.FaceInference;

      if (!faceInference) {
        throw new Error(
          'FaceInference native module is unavailable. Rebuild the Android app.'
        );
      }

      const runtime =
        await faceInference.getRuntimeStatus();

      if (!runtime?.modelsLoaded) {
        const loaded = await faceInference.loadModels();

        if (!loaded?.success) {
          throw new Error(
            loaded?.message ||
              'Unable to load offline face models.'
          );
        }
      }

      // ----------------------------------------------
      // 4. REPAIR MISSING STUDENT EMBEDDINGS
      // USING EXISTING REGISTERED PHOTOGRAPHS
      // ----------------------------------------------

      for (const student of students) {
        if (parseEmbedding(student.embedding)) {
          continue;
        }

        if (!student.photo_uri) {
          console.log(
            'No saved photo:',
            student.registration_number
          );
          continue;
        }

        try {
          const repair =
            await faceInference.extractFaces(
              student.photo_uri
            );

          const embedding =
            repair?.faces?.[0]?.embedding;

          if (
            !repair?.success ||
            repair.faceCount !== 1 ||
            !Array.isArray(embedding) ||
            embedding.length !== 512 ||
            !embedding.every(Number.isFinite)
          ) {
            console.log(
              'Embedding repair unsuccessful:',
              student.registration_number,
              repair?.faceCount
            );

            continue;
          }

          const embeddingJSON =
            JSON.stringify(embedding);

          await db.runAsync(
            `UPDATE students
             SET embedding = ?
             WHERE registration_number = ?`,
            [
              embeddingJSON,
              student.registration_number,
            ]
          );

          student.embedding = embeddingJSON;

          console.log(
            'Embedding repaired:',
            student.registration_number
          );
        } catch (repairError) {
          console.error(
            'Embedding repair failed:',
            student.registration_number,
            repairError
          );
        }
      }

      // ----------------------------------------------
      // 5. PREPARE VALID REFERENCE EMBEDDINGS
      // ----------------------------------------------

      const registeredStudents = students
        .map(student => ({
          ...student,
          vector: parseEmbedding(student.embedding),
        }))
        .filter(student => student.vector !== null);

      const missingEmbeddings =
        students.length - registeredStudents.length;

      console.log(
        'VALID EMBEDDINGS:',
        registeredStudents.length
      );

      console.log(
        'MISSING EMBEDDINGS:',
        missingEmbeddings
      );

      // ----------------------------------------------
      // 6. EXTRACT FACES FROM CLASSROOM IMAGE
      // ----------------------------------------------

      const recognition =
        await faceInference.extractFaces(photo.uri);

      if (!recognition?.success) {
        throw new Error(
          recognition?.message ||
            'Classroom face extraction failed.'
        );
      }

      const detectedFaces =
        recognition.faces || [];

      console.log(
        'DETECTED FACES:',
        detectedFaces.length
      );

      if (!detectedFaces.length) {
        throw new Error(
          'No faces detected in classroom photo.'
        );
      }

      // ----------------------------------------------
      // 7. MATCH DETECTED FACES WITH STUDENTS
      // ----------------------------------------------

      const matchedStudents = new Map();
      const usedStudents = new Set();

      let unknownFaces = 0;

      for (const face of detectedFaces) {
        const faceVector =
          parseEmbedding(face.embedding);

        if (!faceVector) {
          unknownFaces++;
          continue;
        }

        let bestStudent = null;
        let bestScore = -1;

        for (const student of registeredStudents) {
          if (
            usedStudents.has(
              student.registration_number
            )
          ) {
            continue;
          }

          const score = cosineSimilarity(
            faceVector,
            student.vector
          );

          if (score > bestScore) {
            bestScore = score;
            bestStudent = student;
          }
        }

        if (
          bestStudent &&
          bestScore >= SIMILARITY_THRESHOLD
        ) {
          usedStudents.add(
            bestStudent.registration_number
          );

          matchedStudents.set(
            bestStudent.registration_number,
            {
              student: bestStudent,
              score: bestScore,
            }
          );

          console.log(
            'MATCHED:',
            bestStudent.registration_number,
            bestStudent.name,
            bestScore
          );
        } else {
          unknownFaces++;
        }
      }

      // ----------------------------------------------
      // 8. CREATE RECORDS FOR EVERY REGISTERED STUDENT
      // ----------------------------------------------

      const records = students.map(student => {
        const match = matchedStudents.get(
          student.registration_number
        );

        const hasEmbedding =
          !!parseEmbedding(student.embedding);

        return {
          registration_number:
            student.registration_number,

          name: student.name,

          status: match
            ? 'Present'
            : hasEmbedding
              ? 'Absent'
              : 'Embedding Missing',

          similarity_score:
            match ? match.score : null,
        };
      });

      const presentCount = records.filter(
        record => record.status === 'Present'
      ).length;

      const absentCount = records.filter(
        record => record.status === 'Absent'
      ).length;

      const missingCount = records.filter(
        record => record.status === 'Embedding Missing'
      ).length;

      console.log('PRESENT:', presentCount);
      console.log('ABSENT:', absentCount);
      console.log('MISSING:', missingCount);

      // ----------------------------------------------
      // 9. SAVE ATTENDANCE SESSION LOCALLY
      // ----------------------------------------------

      const sessionResult = await db.runAsync(
        `INSERT INTO attendance_sessions
         (
           class_name,
           section,
           course_name,
           course_id,
           class_id,
           slot_no,
           image_uri,
           total_count,
           present_count,
           absent_count
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          `${selectedCourseId} - ${selectedClassId}`,
          null,
          selectedCourseName,
          selectedCourseId,
          selectedClassId,
          selectedSlot,
          photo.uri,
          records.length,
          presentCount,
          absentCount,
        ]
      );

      const sessionId =
        sessionResult.lastInsertRowId;

      // ----------------------------------------------
      // 10. SAVE EVERY STUDENT'S ATTENDANCE RECORD
      // ----------------------------------------------

      for (const record of records) {
        await db.runAsync(
          `INSERT INTO attendance_records
           (
             session_id,
             registration_number,
             student_name,
             status,
             similarity_score
           )
           VALUES (?, ?, ?, ?, ?)`,
          [
            sessionId,
            record.registration_number,
            record.name,
            record.status,
            record.similarity_score,
          ]
        );
      }

      // ----------------------------------------------
      // 11. DISPLAY RESULTS
      // ----------------------------------------------

      setResult({
        course_name: selectedCourseName,
        course_id: selectedCourseId,
        class_id: selectedClassId,
        slot_no: selectedSlot,

        class_session_id: sessionId,

        present_count: presentCount,
        absent_count: absentCount,
        unknown_faces_detected: unknownFaces,
        missing_embeddings: missingCount,

        records: records.map(record => ({
          student: {
            registration_number:
              record.registration_number,
            name: record.name,
          },

          status: record.status,
          similarity_score:
            record.similarity_score,
        })),
      });

      Alert.alert(
        'Offline Attendance Saved',
        `${presentCount} present, ${absentCount} absent.\n` +
          `${unknownFaces} unknown faces.\n` +
          `${missingCount} students have missing embeddings.`
      );
    } catch (error) {
      console.error(
        'Offline attendance error:',
        error
      );

      Alert.alert(
        'Attendance failed',
        error?.message ||
          'Unable to process attendance offline.'
      );
    } finally {
      setLoading(false);
    }
  };

  // ====================================================
  // RENDER INDIVIDUAL ATTENDANCE RECORD
  // ====================================================

  const renderRecord = (record, index) => {
    const student = record.student || {};

    const registrationNumber =
      student.registration_number || 'Unknown';

    const studentName =
      student.name || 'Unknown Student';

    const status =
      record.status || 'Unknown';

    const similarity =
      record.similarity_score;

    const isPresent =
      status.toLowerCase() === 'present';

    return (
      <View
        key={`${registrationNumber}-${index}`}
        style={styles.record}
      >
        <View style={styles.recordInfo}>
          <Text style={styles.registrationNumber}>
            {registrationNumber}
          </Text>

          <Text style={styles.studentName}>
            {studentName}
          </Text>

          {similarity !== null &&
            similarity !== undefined && (
              <Text style={styles.similarity}>
                Similarity:{' '}
                {(similarity * 100).toFixed(1)}%
              </Text>
            )}
        </View>

        <View
          style={[
            styles.statusBadge,
            isPresent
              ? styles.presentBadge
              : status === 'Embedding Missing'
                ? styles.missingBadge
                : styles.absentBadge,
          ]}
        >
          <Text style={styles.statusText}>
            {status}
          </Text>
        </View>
      </View>
    );
  };

  // ====================================================
  // SCREEN UI
  // ====================================================

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.title}>
        Mark Attendance
      </Text>

      <Text style={styles.description}>
        Select the course and class before uploading
        the classroom photo. Attendance will be marked
        only for students registered in this class.
      </Text>

      {/* Course Name */}

      <Text style={styles.label}>
        Course Name
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Enter course name"
        value={courseName}
        onChangeText={setCourseName}
        editable={!loading}
      />

      {/* Course ID */}

      <Text style={styles.label}>
        Course ID
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Enter course ID"
        value={courseId}
        onChangeText={setCourseId}
        autoCapitalize="characters"
        editable={!loading}
      />

      {/* Class ID */}

      <Text style={styles.label}>
        Class ID
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Enter class ID"
        value={classId}
        onChangeText={setClassId}
        autoCapitalize="characters"
        editable={!loading}
      />

      {/* Slot Number */}

      <Text style={styles.label}>
        Slot Number
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Enter slot number"
        value={slotNo}
        onChangeText={setSlotNo}
        autoCapitalize="characters"
        editable={!loading}
      />

      {/* Photo Buttons */}

      <View style={styles.photoButtons}>
        <TouchableOpacity
          style={styles.blueButton}
          onPress={takePhoto}
          disabled={loading}
        >
          <Ionicons
            name="camera"
            size={22}
            color="#fff"
          />

          <Text style={styles.buttonText}>
            Take Photo
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.blueButton}
          onPress={choosePhoto}
          disabled={loading}
        >
          <Ionicons
            name="images"
            size={22}
            color="#fff"
          />

          <Text style={styles.buttonText}>
            Choose Photo
          </Text>
        </TouchableOpacity>
      </View>

      {/* Classroom Image */}

      {photo?.uri && (
        <View style={styles.imageContainer}>
          <Image
            source={{ uri: photo.uri }}
            style={styles.classroomImage}
            resizeMode="contain"
          />

          <TouchableOpacity
            style={styles.removeButton}
            onPress={() => {
              setPhoto(null);
              setResult(null);
            }}
            disabled={loading}
          >
            <Ionicons
              name="close-circle"
              size={30}
              color="#dc2626"
            />
          </TouchableOpacity>
        </View>
      )}

      {/* Mark Attendance Button */}

      <TouchableOpacity
        style={[
          styles.markButton,
          loading && styles.disabledButton,
        ]}
        onPress={markAttendance}
        disabled={loading}
      >
        {loading ? (
          <>
            <ActivityIndicator color="#fff" />

            <Text style={styles.buttonText}>
              Recognizing...
            </Text>
          </>
        ) : (
          <>
            <Ionicons
              name="checkmark-circle"
              size={23}
              color="#fff"
            />

            <Text style={styles.buttonText}>
              Mark Attendance
            </Text>
          </>
        )}
      </TouchableOpacity>

      {/* Attendance Results */}

      {result && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>
            Attendance Results
          </Text>

          <Text style={styles.sessionDetails}>
            Course: {result.course_name || courseName}
          </Text>

          <Text style={styles.sessionDetails}>
            Course ID: {result.course_id || courseId}
          </Text>

          <Text style={styles.sessionDetails}>
            Class ID: {result.class_id || classId}
          </Text>

          {/* Summary */}

          <View style={styles.summary}>
            <View style={styles.summaryBox}>
              <Text style={styles.summaryNumber}>
                {result.present_count ?? 0}
              </Text>

              <Text style={styles.summaryLabel}>
                Present
              </Text>
            </View>

            <View style={styles.summaryBox}>
              <Text style={styles.summaryNumber}>
                {result.absent_count ?? 0}
              </Text>

              <Text style={styles.summaryLabel}>
                Absent
              </Text>
            </View>

            <View style={styles.summaryBox}>
              <Text style={styles.summaryNumber}>
                {result.unknown_faces_detected ?? 0}
              </Text>

              <Text style={styles.summaryLabel}>
                Unknown
              </Text>
            </View>
          </View>

          <Text style={styles.sessionText}>
            Session ID: {result.class_session_id}
          </Text>

          <Text style={styles.sessionText}>
            Missing embeddings: {result.missing_embeddings ?? 0}
          </Text>

          {/* Every Registered Student */}

          {Array.isArray(result.records) &&
            result.records.map(renderRecord)}
        </View>
      )}
    </ScrollView>
  );
}

// ======================================================
// STYLES
// ======================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },

  content: {
    padding: 16,
    paddingBottom: 40,
  },

  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },

  description: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 18,
    lineHeight: 19,
  },

  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 6,
  },

  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 14,
  },

  photoButtons: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 14,
  },

  blueButton: {
    flex: 1,
    backgroundColor: '#2563eb',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },

  buttonText: {
    color: '#fff',
    fontWeight: '600',
    marginLeft: 8,
  },

  imageContainer: {
    position: 'relative',
    backgroundColor: '#fff',
    borderRadius: 10,
    marginBottom: 14,
    overflow: 'hidden',
  },

  classroomImage: {
    width: '100%',
    height: 300,
  },

  removeButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: '#fff',
    borderRadius: 20,
  },

  markButton: {
    backgroundColor: '#16a34a',
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    marginBottom: 20,
  },

  disabledButton: {
    opacity: 0.6,
  },

  resultsContainer: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 14,
  },

  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 14,
  },

  sessionDetails: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 5,
  },

  summary: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
    marginTop: 10,
  },

  summaryBox: {
    flex: 1,
    alignItems: 'center',
    padding: 10,
    marginHorizontal: 3,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
  },

  summaryNumber: {
    fontSize: 24,
    fontWeight: 'bold',
  },

  summaryLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 3,
  },

  sessionText: {
    color: '#6b7280',
    marginBottom: 12,
  },

  record: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    marginBottom: 8,
  },

  recordInfo: {
    flex: 1,
  },

  registrationNumber: {
    fontSize: 15,
    fontWeight: 'bold',
  },

  studentName: {
    fontSize: 14,
    color: '#4b5563',
    marginTop: 2,
  },

  similarity: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 3,
  },

  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 15,
  },

  presentBadge: {
    backgroundColor: '#dcfce7',
  },

  absentBadge: {
    backgroundColor: '#fee2e2',
  },

  missingBadge: {
    backgroundColor: '#fef3c7',
  },

  statusText: {
    fontWeight: '600',
  },
});