import React, { useCallback, useState } from 'react';
import { NativeModules } from 'react-native';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  Alert,
  StyleSheet,
  ActivityIndicator,
  Image,
} from 'react-native';

import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

import { getDatabase } from '../database';

export default function RegisterScreen() {
  const [regNo, setRegNo] = useState('');
  const [name, setName] = useState('');
  const [photo, setPhoto] = useState(null);
  const [saving, setSaving] = useState(false);
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  // Test native AI model initialization
const testAIModels = async () => {
  if (!NativeModules.FaceInference) {
    Alert.alert(
      'Native Module Error',
      'FaceInference is not available. Rebuild the Android app.'
    );
    return;
  }

  setAiLoading(true);

  try {
    const status = await NativeModules.FaceInference.getRuntimeStatus();

    if (status.modelsLoaded) {
      Alert.alert('AI Ready', 'All AI models are already loaded.');
      return;
    }

    const result = await NativeModules.FaceInference.loadModels();

    Alert.alert(
      'AI Models Loaded',
      `${result.message}\n\nYOLO: ${result.yolo}\nRetinaFace: ${result.faceDetector}\nArcFace: ${result.arcface}\nOpenCV: ${result.openCVReady ? 'Ready' : 'Not Ready'}`
    );
  } catch (error) {
    console.error('AI initialization failed:', error);

    Alert.alert(
      'AI Initialization Failed',
      error.message || 'Unable to load AI models.'
    );
  } finally {
    setAiLoading(false);
  }
};
  // Load students from SQLite
  const loadStudents = useCallback(async () => {
    setLoading(true);

    try {
      const db = await getDatabase();

      const result = await db.getAllAsync(
        `SELECT registration_number, name, department,
                year, section, photo_uri
         FROM students
         ORDER BY created_at DESC`
      );

      setStudents(result);
    } catch (error) {
      Alert.alert('Database Error', error.message);
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    loadStudents();
  }, [loadStudents]);

  // Camera or gallery selection
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

  // Take photo
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

  // Select photo from gallery
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

  // Copy photo into permanent app storage
  const savePhotoLocally = async (uri, registrationNumber) => {
    const fileName =
      `${registrationNumber.replace(/[^a-zA-Z0-9_-]/g, '_')}_${Date.now()}.jpg`;

    const destination = `${FileSystem.documentDirectory}${fileName}`;

    await FileSystem.copyAsync({
      from: uri,
      to: destination,
    });

    return destination;
  };

  // Register student locally
  const handleRegister = async () => {
    const registrationNumber = regNo.trim().toUpperCase();
    const studentName = name.trim();

    if (!registrationNumber || !studentName) {
      Alert.alert(
        'Validation',
        'Registration number and name are required.'
      );
      return;
    }

    if (!photo?.uri) {
      Alert.alert(
        'Photo required',
        'Please take a photo or choose a photo from the gallery.'
      );
      return;
    }

    setSaving(true);

    let savedPhotoUri = null;

    try {
      const db = await getDatabase();

      const existing = await db.getFirstAsync(
        'SELECT registration_number FROM students WHERE registration_number = ?',
        registrationNumber
      );

      if (existing) {
        Alert.alert(
          'Already registered',
          'This registration number already exists.'
        );
        return;
      }

      savedPhotoUri = await savePhotoLocally(
        photo.uri,
        registrationNumber
      );

      await db.runAsync(
        `INSERT INTO students
         (registration_number, name, department, year,
          section, photo_uri)
         VALUES (?, ?, ?, ?, ?, ?)`,
        registrationNumber,
        studentName,
        'ECE',
        3,
        'A',
        savedPhotoUri
      );

      Alert.alert(
        'Success',
        `Student "${studentName}" registered successfully.`
      );

      setRegNo('');
      setName('');
      setPhoto(null);

      await loadStudents();
    } catch (error) {
      if (savedPhotoUri) {
        await FileSystem.deleteAsync(savedPhotoUri, {
          idempotent: true,
        }).catch(() => {});
      }

      Alert.alert(
        'Registration failed',
        error.message || 'Unable to save student.'
      );
    } finally {
      setSaving(false);
    }
  };

  // Confirm deletion
  const confirmDelete = (student) => {
    Alert.alert(
      'Delete student',
      `Delete ${student.name} (${student.registration_number})? This cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () =>
            deleteStudent(student.registration_number),
        },
      ]
    );
  };

  // Delete student locally
  const deleteStudent = async (registrationNumber) => {
    try {
      const db = await getDatabase();

      const student = await db.getFirstAsync(
        'SELECT photo_uri FROM students WHERE registration_number = ?',
        registrationNumber
      );

      await db.runAsync(
        'DELETE FROM students WHERE registration_number = ?',
        registrationNumber
      );

      if (student?.photo_uri) {
        await FileSystem.deleteAsync(student.photo_uri, {
          idempotent: true,
        }).catch(() => {});
      }

      await loadStudents();

      Alert.alert('Success', 'Student deleted successfully.');
    } catch (error) {
      Alert.alert('Delete failed', error.message);
    }
  };

  // Student list item
  const renderItem = ({ item }) => (
    <View style={styles.row}>
      <View style={styles.rowText}>
        <Text style={styles.regNo}>
          {item.registration_number}
        </Text>

        <Text style={styles.studentName}>
          {item.name}
        </Text>

        {(item.department || item.year || item.section) && (
          <Text style={styles.details}>
            {item.department || ''}
            {item.year ? ` • Year ${item.year}` : ''}
            {item.section ? ` • Section ${item.section}` : ''}
          </Text>
        )}
      </View>

      <TouchableOpacity
        style={styles.deleteBtn}
        onPress={() => confirmDelete(item)}
        hitSlop={{
          top: 10,
          bottom: 10,
          left: 10,
          right: 10,
        }}
      >
        <Ionicons name="trash" size={22} color="#dc2626" />
      </TouchableOpacity>
    </View>
  );

  // Screen
  return (
    <View style={styles.container}>
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
          {photo ? 'Change Student Photo' : 'Take / Choose Student Photo'}
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
              Registering...
            </Text>
          </>
        ) : (
          <Text style={styles.buttonText}>
            Register
          </Text>
        )}
      </TouchableOpacity>

      <View style={styles.listHeader}>
        <Text style={styles.sectionTitle}>
          Students
        </Text>

        <TouchableOpacity
          onPress={loadStudents}
          disabled={loading}
        >
          <Ionicons
            name="reload"
            size={22}
            color="#2563eb"
          />
        </TouchableOpacity>
      </View>

      <FlatList
        data={students}
        keyExtractor={(item) => String(item.registration_number)}
        renderItem={renderItem}
        ListEmptyComponent={
          <Text style={styles.empty}>
            {loading ? 'Loading…' : 'No students yet.'}
          </Text>
        }
        contentContainerStyle={styles.listContent}
      />
    </View>
  );
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    paddingTop: 40,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
    fontSize: 16,
    backgroundColor: '#fff',
  },
  photoButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#2563eb',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
  },
  photoButtonText: {
    color: '#2563eb',
    fontWeight: '600',
    marginLeft: 8,
  },
  previewContainer: {
    alignItems: 'center',
    marginBottom: 12,
    position: 'relative',
  },
  preview: {
    width: 120,
    height: 120,
    borderRadius: 10,
  },
  removePhoto: {
    position: 'absolute',
    right: '31%',
    top: -8,
    backgroundColor: '#fff',
    borderRadius: 20,
  },
  button: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    minHeight: 48,
    flexDirection: 'row',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
    marginLeft: 8,
  },
  listHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    marginBottom: 6,
    backgroundColor: '#fff',
  },
  rowText: {
    flex: 1,
  },
  regNo: {
    fontWeight: '600',
    fontSize: 15,
  },
  studentName: {
    color: '#4b5563',
    fontSize: 15,
    marginTop: 2,
  },
  details: {
    color: '#6b7280',
    fontSize: 12,
    marginTop: 3,
  },
  deleteBtn: {
    padding: 6,
  },
  empty: {
    textAlign: 'center',
    color: '#6b7280',
    marginTop: 16,
  },
  listContent: {
    paddingBottom: 24,
  },
});
