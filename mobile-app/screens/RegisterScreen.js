
import React, { useCallback, useState } from 'react';

import {
  NativeModules,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  StyleSheet,
  ActivityIndicator,
  Image,
  ScrollView,
} from 'react-native';

import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

import { getDatabase } from '../database';

export default function RegisterScreen() {
  const [regNo, setRegNo] = useState('');
  const [name, setName] = useState('');
  const [courseName, setCourseName] = useState('');
  const [courseId, setCourseId] = useState('');
  const [classId, setClassId] = useState('');
  const [slotNo, setSlotNo] = useState('');

  const [photo, setPhoto] = useState(null);
  const [saving, setSaving] = useState(false);

  // Load students from local SQLite only.
  // No backend request is made.
  const loadStudents = useCallback(async () => {
    try {
      const db = await getDatabase();

      const students = await db.getAllAsync(
        `SELECT registration_number, name,
                course_name, course_id, class_id,
                slot_no, photo_uri
         FROM students
         ORDER BY created_at DESC`
      );

      console.log(
        'LOCAL STUDENTS:',
        students.map(student => student.registration_number)
      );

      console.log('Offline student loading successful.');
    } catch (error) {
      console.error('Database loading failed:', error);
      Alert.alert('Database Error', error.message);
    }
  }, []);

  React.useEffect(() => {
    loadStudents();
  }, [loadStudents]);

  // Select camera or gallery.
  const choosePhoto = () => {
    Alert.alert(
      'Select Student Photo',
      'Choose how you want to add the student photo.',
      [
        { text: 'Camera', onPress: takePhoto },
        { text: 'Gallery', onPress: pickFromGallery },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  // Capture photo.
  const takePhoto = async () => {
    try {
      const permission =
        await ImagePicker.requestCameraPermissionsAsync();

      if (!permission.granted) {
        Alert.alert(
          'Permission required',
          'Camera permission is required.'
        );
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets?.length > 0) {
        setPhoto(result.assets[0]);
      }
    } catch (error) {
      Alert.alert('Camera error', error.message);
    }
  };

  // Select photo from gallery.
  const pickFromGallery = async () => {
    try {
      const permission =
        await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (!permission.granted) {
        Alert.alert(
          'Permission required',
          'Gallery permission is required.'
        );
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets?.length > 0) {
        setPhoto(result.assets[0]);
      }
    } catch (error) {
      Alert.alert('Gallery error', error.message);
    }
  };

  // Copy photo to permanent local app storage.
  const savePhotoLocally = async (uri, registrationNumber) => {
    const fileName =
      `${registrationNumber.replace(/[^a-zA-Z0-9_-]/g, '_')}_${Date.now()}.jpg`;

    const destination =
      `${FileSystem.documentDirectory}${fileName}`;

    await FileSystem.copyAsync({
      from: uri,
      to: destination,
    });

    return destination;
  };

  // Generate and validate a face embedding.
  const generateEmbedding = async (imageUri) => {
    const faceModule = NativeModules.FaceInference;

    if (!faceModule) {
      throw new Error(
        'FaceInference native module is unavailable. Rebuild the Android app.'
      );
    }

    const status = await faceModule.getRuntimeStatus();

    if (!status?.modelsLoaded) {
      await faceModule.loadModels();
    }

    const result = await faceModule.extractFaces(imageUri);
    console.log('FULL NATIVE RESULT:', JSON.stringify(result));

    console.log(
      'REGISTRATION FACE EXTRACTION:',
      JSON.stringify({
        success: result?.success,
        faceCount: result?.faceCount,
        returnedFaces: result?.faces?.length,
      })
    );

    if (!result?.success || !result.faces?.length) {
      throw new Error(
        'No face detected. Please select a clear individual student photo.'
      );
    }

    if (result.faces.length !== 1) {
      throw new Error(
        `Exactly one face is required. Detected ${result.faces.length} faces.`
      );
    }

    const embedding = result.faces[0].embedding;

    if (
      !Array.isArray(embedding) ||
      embedding.length !== 512 ||
      embedding.some(value => !Number.isFinite(value))
    ) {
      throw new Error(
        'Invalid face embedding. Expected 512 valid numerical values.'
      );
    }

    return embedding;
  };

  // Register a new student or repair an existing student's embedding.
  const handleRegister = async () => {
    const registrationNumber = regNo.trim().toUpperCase();
    const studentName = name.trim();
    const studentCourseName = courseName.trim();
    const studentCourseId = courseId.trim().toUpperCase();
    const studentClassId = classId.trim().toUpperCase();
    const studentSlotNo = slotNo.trim().toUpperCase();

    if (
      !registrationNumber ||
      !studentName ||
      !studentCourseName ||
      !studentCourseId ||
      !studentClassId ||
      !studentSlotNo
    ) {
      Alert.alert(
        'Validation',
        'Please fill in all six registration fields.'
      );
      return;
    }

    if (!/^[A-Za-z0-9_-]+$/.test(registrationNumber)) {
      Alert.alert(
        'Invalid Registration Number',
        'Use only letters, numbers, hyphens and underscores.'
      );
      return;
    }

    if (!photo?.uri) {
      Alert.alert(
        'Photo required',
        'Please select an individual student photo.'
      );
      return;
    }

    setSaving(true);

    let savedPhotoUri = null;
    let databaseUpdated = false;

    try {
      const db = await getDatabase();

      // Check the existing registration before modifying anything.
      const existing = await db.getFirstAsync(
        `SELECT registration_number, photo_uri
         FROM students
         WHERE registration_number = ?`,
        registrationNumber
      );

      // Generate embedding locally using ONNX models.
      const embedding = await generateEmbedding(photo.uri);

      console.log(
        'EMBEDDING GENERATED:',
        registrationNumber,
        embedding.length
      );

      // Save selected photo locally.
      savedPhotoUri = await savePhotoLocally(
        photo.uri,
        registrationNumber
      );

      const embeddingJSON = JSON.stringify(embedding);

      if (existing) {
        // Update the existing record. Do not insert a duplicate.
        const updateResult = await db.runAsync(
          `UPDATE students
           SET name = ?,
               course_name = ?,
               course_id = ?,
               class_id = ?,
               slot_no = ?,
               photo_uri = ?,
               embedding = ?
           WHERE registration_number = ?`,
          studentName,
          studentCourseName,
          studentCourseId,
          studentClassId,
          studentSlotNo,
          savedPhotoUri,
          embeddingJSON,
          registrationNumber
        );

        if (updateResult.changes !== 1) {
          throw new Error(
            'The existing student record could not be updated.'
          );
        }

        databaseUpdated = true;

        console.log(
          'EXISTING STUDENT EMBEDDING UPDATED:',
          registrationNumber
        );
      } else {
        // Insert only when this registration number does not exist.
        await db.runAsync(
          `INSERT INTO students
           (registration_number, name, course_name, course_id,
            class_id, slot_no, photo_uri, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
          registrationNumber,
          studentName,
          studentCourseName,
          studentCourseId,
          studentClassId,
          studentSlotNo,
          savedPhotoUri,
          embeddingJSON
        );

        databaseUpdated = true;

        console.log(
          'NEW STUDENT REGISTERED:',
          registrationNumber
        );
      }

      // Verify that the embedding was saved.
      const verification = await db.getFirstAsync(
        `SELECT registration_number, embedding
         FROM students
         WHERE registration_number = ?`,
        registrationNumber
      );

      if (!verification?.embedding) {
        throw new Error(
          'Embedding was not found after saving.'
        );
      }

      const savedEmbedding =
        typeof verification.embedding === 'string'
          ? JSON.parse(verification.embedding)
          : verification.embedding;

      if (
        !Array.isArray(savedEmbedding) ||
        savedEmbedding.length !== 512
      ) {
        throw new Error(
          'Saved embedding verification failed.'
        );
      }

      console.log(
        'EMBEDDING VERIFIED:',
        registrationNumber,
        savedEmbedding.length
      );

      // Refresh local data.
      await loadStudents();

      // Reset form after successful save.
      setRegNo('');
      setName('');
      setCourseName('');
      setCourseId('');
      setClassId('');
      setSlotNo('');
      setPhoto(null);

      Alert.alert(
        'Success',
        existing
          ? `${studentName}'s existing registration was updated and the face embedding was saved locally.`
          : `${studentName} was registered locally with a face embedding.`
      );

    } catch (error) {
      console.error(
        'REGISTRATION / EMBEDDING ERROR:',
        error
      );

      // Only remove the newly copied photo if the database
      // was not updated. Never delete an existing student's photo.
      if (savedPhotoUri && !databaseUpdated) {
        await FileSystem.deleteAsync(savedPhotoUri, {
          idempotent: true,
        }).catch(() => {});
      }

      Alert.alert(
        'Registration Failed',
        error.message || 'Unable to register student.'
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.screenContent}
      keyboardShouldPersistTaps="handled"
      showsVerticalScrollIndicator={true}
    >
      <Text style={styles.sectionTitle}>
        Register Student
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Registration No."
        autoCapitalize="characters"
        value={regNo}
        onChangeText={setRegNo}
        editable={!saving}
      />

      <TextInput
        style={styles.input}
        placeholder="Full Name"
        value={name}
        onChangeText={setName}
        editable={!saving}
      />

      <TextInput
        style={styles.input}
        placeholder="Course Name"
        value={courseName}
        onChangeText={setCourseName}
        editable={!saving}
      />

      <TextInput
        style={styles.input}
        placeholder="Course ID"
        autoCapitalize="characters"
        value={courseId}
        onChangeText={setCourseId}
        editable={!saving}
      />

      <TextInput
        style={styles.input}
        placeholder="Class ID"
        autoCapitalize="characters"
        value={classId}
        onChangeText={setClassId}
        editable={!saving}
      />

      <TextInput
        style={styles.input}
        placeholder="Slot Number"
        autoCapitalize="characters"
        value={slotNo}
        onChangeText={setSlotNo}
        editable={!saving}
      />

      <TouchableOpacity
        style={styles.photoButton}
        onPress={choosePhoto}
        disabled={saving}
      >
        <Ionicons
          name="camera-outline"
          size={22}
          color="#2563eb"
        />

        <Text style={styles.photoButtonText}>
          {photo
            ? 'Change Student Photo'
            : 'Take / Choose Student Photo'}
        </Text>
      </TouchableOpacity>

      {photo?.uri && (
        <View style={styles.previewContainer}>
          <Image
            source={{ uri: photo.uri }}
            style={styles.preview}
          />

          <TouchableOpacity
            style={styles.removePhoto}
            onPress={() => setPhoto(null)}
            disabled={saving}
          >
            <Ionicons
              name="close-circle"
              size={28}
              color="#dc2626"
            />
          </TouchableOpacity>
        </View>
      )}

      <TouchableOpacity
        style={[
          styles.button,
          saving && styles.buttonDisabled,
        ]}
        onPress={handleRegister}
        disabled={saving}
      >
        {saving ? (
          <>
            <ActivityIndicator color="#fff" />

            <Text style={styles.buttonText}>
              Processing Face...
            </Text>
          </>
        ) : (
          <Text style={styles.buttonText}>
            Register / Update Student
          </Text>
        )}
      </TouchableOpacity>

      <Text style={styles.infoText}>
        Student information, photos and face embeddings
        are saved locally on this device.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },

  screenContent: {
    padding: 16,
    paddingTop: 40,
    paddingBottom: 40,
  },

  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#111827',
  },

  input: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    fontSize: 16,
    backgroundColor: '#fff',
    color: '#111827',
  },

  photoButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#2563eb',
    borderRadius: 8,
    padding: 14,
    marginBottom: 14,
    backgroundColor: '#fff',
  },

  photoButtonText: {
    color: '#2563eb',
    fontWeight: '600',
    marginLeft: 8,
  },

  previewContainer: {
    alignItems: 'center',
    marginBottom: 16,
    position: 'relative',
  },

  preview: {
    width: 140,
    height: 140,
    borderRadius: 10,
  },

  removePhoto: {
    position: 'absolute',
    right: '30%',
    top: -8,
    backgroundColor: '#fff',
    borderRadius: 20,
  },

  button: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    minHeight: 50,
    flexDirection: 'row',
    gap: 8,
  },

  buttonDisabled: {
    opacity: 0.6,
  },

  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },

  infoText: {
    textAlign: 'center',
    color: '#6b7280',
    fontSize: 13,
    marginTop: 8,
  },
});
