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
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import api, { getApiErrorMessage } from '../api';

export default function AttendanceScreen() {
  const [photo, setPhoto] = useState(null);
  const [className, setClassName] = useState('');
  const [section, setSection] = useState('A');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // ------------------------------------------------------------
  // Take classroom photo
  // ------------------------------------------------------------
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

      const result =
        await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: false,
          quality: 0.8,
        });

      if (!result.canceled && result.assets?.length > 0) {
        setPhoto(result.assets[0]);
        setResult(null);
      }
    } catch (error) {
      Alert.alert(
        'Camera error',
        error?.message || 'Unable to open camera.'
      );
    }
  };

  // ------------------------------------------------------------
  // Choose classroom photo from gallery
  // ------------------------------------------------------------
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

      const result =
        await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: false,
          quality: 0.8,
        });

      if (!result.canceled && result.assets?.length > 0) {
        setPhoto(result.assets[0]);
        setResult(null);
      }
    } catch (error) {
      Alert.alert(
        'Gallery error',
        error?.message || 'Unable to open gallery.'
      );
    }
  };

  // ------------------------------------------------------------
  // Mark attendance
  // ------------------------------------------------------------
  const markAttendance = async () => {
    if (!photo?.uri) {
      Alert.alert(
        'Photo required',
        'Please take or choose a classroom photo first.'
      );
      return;
    }

    if (!className.trim()) {
      Alert.alert(
        'Class required',
        'Please enter the class name.'
      );
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();

      // Backend requires class_name
      formData.append(
        'class_name',
        className.trim()
      );

      // Backend accepts section
      formData.append(
        'section',
        section.trim()
      );

      // Backend requires image
      formData.append('image', {
        uri: photo.uri,
        name: 'classroom.jpg',
        type: photo.mimeType || 'image/jpeg',
      });
      const response = await api.post(
      '/recognize-and-mark/',
      formData
      );
      console.log('ATTENDANCE API RESPONSE:', JSON.stringify(response.data));
      setResult(response.data);
      
      Alert.alert(
        'Attendance Complete',
        `${response.data.present_count} student(s) marked present.`
      );
    } catch (err) {
      Alert.alert(
        'Attendance failed',
        getApiErrorMessage(
          err,
          'Failed to recognize students and mark attendance.'
        )
      );
    } finally {
      setLoading(false);
    }
  };

  // ------------------------------------------------------------
  // Render attendance record
  // ------------------------------------------------------------
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

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
    >
      <Text style={styles.title}>
        Mark Attendance
      </Text>

      {/* Class name */}
      <Text style={styles.label}>
        Class Name
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Enter class name"
        value={className}
        onChangeText={setClassName}
        editable={!loading}
      />

      {/* Section */}
      <Text style={styles.label}>
        Section
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Enter section"
        value={section}
        onChangeText={setSection}
        autoCapitalize="characters"
        editable={!loading}
      />

      {/* Photo buttons */}
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

      {/* Classroom image */}
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

      {/* Mark attendance */}
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

      {/* Results */}
      {result && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>
            Attendance Results
          </Text>

          <View style={styles.summary}>
            <View style={styles.summaryBox}>
              <Text style={styles.summaryNumber}>
                {result.present_count}
              </Text>

              <Text style={styles.summaryLabel}>
                Present
              </Text>
            </View>

            <View style={styles.summaryBox}>
              <Text style={styles.summaryNumber}>
                {result.absent_count}
              </Text>

              <Text style={styles.summaryLabel}>
                Absent
              </Text>
            </View>

            <View style={styles.summaryBox}>
              <Text style={styles.summaryNumber}>
                {result.unknown_faces_detected}
              </Text>

              <Text style={styles.summaryLabel}>
                Unknown
              </Text>
            </View>
          </View>

          <Text style={styles.sessionText}>
            Session ID: {result.class_session_id}
          </Text>

          {Array.isArray(result.records) &&
            result.records.map(renderRecord)}
        </View>
      )}
    </ScrollView>
  );
}

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
    marginBottom: 20,
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

  summary: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
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

  statusText: {
    fontWeight: '600',
  },
});