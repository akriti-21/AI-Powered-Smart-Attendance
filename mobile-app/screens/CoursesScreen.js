
import React, { useCallback, useState } from 'react';
import { useFocusEffect } from '@react-navigation/native';

import {
  View,
  Text,
  TextInput,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  Alert,
  TouchableOpacity,
  Image,
} from 'react-native';

import { getDatabase } from '../database';

export default function CoursesScreen() {
  const [courses, setCourses] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);

  const [selectedCourse, setSelectedCourse] = useState(null);
  const [classSlots, setClassSlots] = useState([]);
  const [loadingClasses, setLoadingClasses] = useState(false);

  const [selectedClassSlot, setSelectedClassSlot] = useState(null);
  const [registeredStudents, setRegisteredStudents] = useState([]);
  const [loadingStudents, setLoadingStudents] = useState(false);

  const [deletingStudent, setDeletingStudent] = useState(null);

  // Load all registered courses
  const loadCourses = async () => {
    try {
      const db = await getDatabase();

      const result = await db.getAllAsync(`
        SELECT
          course_name,
          course_id,
          COUNT(*) AS student_count
        FROM students
        WHERE course_name IS NOT NULL
          AND course_id IS NOT NULL
        GROUP BY course_name, course_id
        ORDER BY course_name COLLATE NOCASE
      `);

      setCourses(result);
    } catch (error) {
      console.error('Courses loading failed:', error);
      Alert.alert('Error', 'Unable to load registered courses.');
    } finally {
      setLoading(false);
    }
  };

  useFocusEffect(
    useCallback(() => {
      loadCourses();
    }, [])
  );

  // Load classes and slots for selected course
  const loadClassSlots = async (course) => {
    setSelectedCourse(course);
    setSelectedClassSlot(null);
    setRegisteredStudents([]);
    setLoadingClasses(true);
    setClassSlots([]);

    try {
      const db = await getDatabase();

      const result = await db.getAllAsync(
        `
        SELECT
          class_id,
          slot_no,
          COUNT(*) AS student_count
        FROM students
        WHERE course_name = ?
          AND course_id = ?
          AND class_id IS NOT NULL
          AND slot_no IS NOT NULL
        GROUP BY class_id, slot_no
        ORDER BY class_id, slot_no
        `,
        course.course_name,
        course.course_id
      );

      setClassSlots(result);
    } catch (error) {
      console.error('Class loading failed:', error);
      Alert.alert('Error', 'Unable to load classes.');
    } finally {
      setLoadingClasses(false);
    }
  };

  // Load students for selected class and slot
  const loadRegisteredStudents = async (classSlot) => {
    setSelectedClassSlot(classSlot);
    setLoadingStudents(true);
    setRegisteredStudents([]);

    try {
      const db = await getDatabase();

      const result = await db.getAllAsync(
        `
        SELECT
          registration_number,
          name,
          course_name,
          course_id,
          class_id,
          slot_no,
          photo_uri
        FROM students
        WHERE course_name = ?
          AND course_id = ?
          AND class_id = ?
          AND slot_no = ?
        ORDER BY name COLLATE NOCASE
        `,
        selectedCourse.course_name,
        selectedCourse.course_id,
        classSlot.class_id,
        classSlot.slot_no
      );

      setRegisteredStudents(result);
    } catch (error) {
      console.error('Student loading failed:', error);
      Alert.alert('Error', 'Unable to load registered students.');
    } finally {
      setLoadingStudents(false);
    }
  };

  // Refresh all displayed data after deletion
  const refreshAfterDelete = async () => {
    const db = await getDatabase();

    // Refresh courses and course counts
    const updatedCourses = await db.getAllAsync(`
      SELECT
        course_name,
        course_id,
        COUNT(*) AS student_count
      FROM students
      WHERE course_name IS NOT NULL
        AND course_id IS NOT NULL
      GROUP BY course_name, course_id
      ORDER BY course_name COLLATE NOCASE
    `);

    setCourses(updatedCourses);

    // Refresh class and slot counts
    if (selectedCourse) {
      const updatedClasses = await db.getAllAsync(
        `
        SELECT
          class_id,
          slot_no,
          COUNT(*) AS student_count
        FROM students
        WHERE course_name = ?
          AND course_id = ?
          AND class_id IS NOT NULL
          AND slot_no IS NOT NULL
        GROUP BY class_id, slot_no
        ORDER BY class_id, slot_no
        `,
        selectedCourse.course_name,
        selectedCourse.course_id
      );

      setClassSlots(updatedClasses);
    }

    // Refresh registered students
    if (selectedCourse && selectedClassSlot) {
      const updatedStudents = await db.getAllAsync(
        `
        SELECT
          registration_number,
          name,
          course_name,
          course_id,
          class_id,
          slot_no,
          photo_uri
        FROM students
        WHERE course_name = ?
          AND course_id = ?
          AND class_id = ?
          AND slot_no = ?
        ORDER BY name COLLATE NOCASE
        `,
        selectedCourse.course_name,
        selectedCourse.course_id,
        selectedClassSlot.class_id,
        selectedClassSlot.slot_no
      );

      setRegisteredStudents(updatedStudents);
    }
  };

  // Delete student from local database
  const deleteStudent = async (student) => {
    if (deletingStudent) return;

    setDeletingStudent(student.registration_number);

    try {
      const db = await getDatabase();

      const result = await db.runAsync(
        `
        DELETE FROM students
        WHERE registration_number = ?
          AND course_name = ?
          AND course_id = ?
          AND class_id = ?
          AND slot_no = ?
        `,
        student.registration_number,
        selectedCourse.course_name,
        selectedCourse.course_id,
        selectedClassSlot.class_id,
        selectedClassSlot.slot_no
      );

      if (result.changes === 0) {
        throw new Error('Student not found. Please refresh the screen.');
      }

      console.log(
        'Student deleted:',
        student.registration_number
      );

      await refreshAfterDelete();

      Alert.alert(
        'Deleted successfully',
        `${student.name} has been removed from the registered students.`
      );
    } catch (error) {
      console.error('Student deletion failed:', error);

      Alert.alert(
        'Delete failed',
        error.message || 'Unable to delete student.'
      );
    } finally {
      setDeletingStudent(null);
    }
  };

  // Ask for confirmation before deleting
  const confirmDelete = (student) => {
    Alert.alert(
      'Delete student?',
      `Are you sure you want to delete ${student.name} (${student.registration_number})?\n\nThis removes their student registration and saved face embedding. Existing attendance history is retained.`,
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => deleteStudent(student),
        },
      ]
    );
  };

  // Search courses
  const filteredCourses = courses.filter(course =>
    `${course.course_name} ${course.course_id}`
      .toLowerCase()
      .includes(search.toLowerCase().trim())
  );

  // Render course card
  const renderCourse = ({ item }) => (
    <TouchableOpacity
      style={styles.courseCard}
      activeOpacity={0.7}
      onPress={() => loadClassSlots(item)}
    >
      <View style={styles.courseHeader}>
        <Text style={styles.courseName}>
          {item.course_name}
        </Text>

        <Text style={styles.arrow}>›</Text>
      </View>

      <Text style={styles.courseId}>
        Course ID: {item.course_id}
      </Text>

      <Text style={styles.studentCount}>
        {item.student_count} registered student(s)
      </Text>
    </TouchableOpacity>
  );

  // Render class and slot card
  const renderClassSlot = ({ item }) => (
    <TouchableOpacity
      style={styles.classCard}
      activeOpacity={0.7}
      onPress={() => loadRegisteredStudents(item)}
    >
      <View style={styles.courseHeader}>
        <Text style={styles.classTitle}>
          Class ID: {item.class_id}
        </Text>

        <Text style={styles.arrow}>›</Text>
      </View>

      <Text style={styles.slotText}>
        Slot Number: {item.slot_no}
      </Text>

      <Text style={styles.studentCount}>
        {item.student_count} registered student(s)
      </Text>
    </TouchableOpacity>
  );

  // Render registered student with Delete option
  const renderStudent = ({ item }) => {
    const isDeleting =
      deletingStudent === item.registration_number;

    return (
      <View style={styles.studentCard}>
        {item.photo_uri ? (
          <Image
            source={{ uri: item.photo_uri }}
            style={styles.studentPhoto}
          />
        ) : (
          <View style={styles.photoPlaceholder}>
            <Text style={styles.photoPlaceholderText}>
              {item.name?.charAt(0)?.toUpperCase() || '?'}
            </Text>
          </View>
        )}

        <View style={styles.studentInfo}>
          <Text style={styles.studentName}>
            {item.name}
          </Text>

          <Text style={styles.registrationNumber}>
            {item.registration_number}
          </Text>

          <Text style={styles.studentSlot}>
            {item.class_id} · Slot {item.slot_no}
          </Text>
        </View>

        <TouchableOpacity
          style={[
            styles.deleteButton,
            isDeleting && styles.disabledDeleteButton,
          ]}
          activeOpacity={0.7}
          disabled={!!deletingStudent}
          onPress={() => confirmDelete(item)}
        >
          {isDeleting ? (
            <ActivityIndicator
              size="small"
              color="#ffffff"
            />
          ) : (
            <Text style={styles.deleteButtonText}>
              Delete
            </Text>
          )}
        </TouchableOpacity>
      </View>
    );
  };

  return (
    <View style={styles.container}>

      {/* SCREEN 1: COURSES */}

      {selectedCourse === null ? (
        <>
          <Text style={styles.title}>
            Courses
          </Text>

          <Text style={styles.subtitle}>
            View your registered courses and students.
          </Text>

          <TextInput
            style={styles.searchInput}
            placeholder="Search by Course Name or Course ID"
            placeholderTextColor="#9ca3af"
            value={search}
            onChangeText={setSearch}
            autoCapitalize="none"
          />

          {loading ? (
            <ActivityIndicator
              size="large"
              color="#2563eb"
              style={styles.loader}
            />
          ) : (
            <FlatList
              data={filteredCourses}
              keyExtractor={item =>
                `${item.course_id}-${item.course_name}`
              }
              renderItem={renderCourse}
              contentContainerStyle={styles.listContent}
              showsVerticalScrollIndicator={false}
              ListEmptyComponent={
                <Text style={styles.emptyText}>
                  {search.trim()
                    ? 'No matching courses found.'
                    : 'No registered courses found.'}
                </Text>
              }
            />
          )}
        </>

      ) : selectedClassSlot === null ? (

        /* SCREEN 2: CLASSES AND SLOTS */

        <>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => setSelectedCourse(null)}
          >
            <Text style={styles.backText}>
              ‹ Back to Courses
            </Text>
          </TouchableOpacity>

          <Text style={styles.title}>
            {selectedCourse.course_name}
          </Text>

          <Text style={styles.courseId}>
            Course ID: {selectedCourse.course_id}
          </Text>

          <Text style={styles.subtitle}>
            Registered Classes and Slots
          </Text>

          {loadingClasses ? (
            <ActivityIndicator
              size="large"
              color="#2563eb"
              style={styles.loader}
            />
          ) : (
            <FlatList
              data={classSlots}
              keyExtractor={(item, index) =>
                `${item.class_id}-${item.slot_no}-${index}`
              }
              renderItem={renderClassSlot}
              contentContainerStyle={styles.listContent}
              showsVerticalScrollIndicator={false}
              ListEmptyComponent={
                <Text style={styles.emptyText}>
                  No registered classes found.
                </Text>
              }
            />
          )}
        </>

      ) : (

        /* SCREEN 3: REGISTERED STUDENTS */

        <>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => setSelectedClassSlot(null)}
          >
            <Text style={styles.backText}>
              ‹ Back to Classes
            </Text>
          </TouchableOpacity>

          <Text style={styles.title}>
            Registered Students
          </Text>

          <Text style={styles.courseId}>
            {selectedCourse.course_name}
          </Text>

          <View style={styles.selectedClassCard}>
            <Text style={styles.classTitle}>
              Class ID: {selectedClassSlot.class_id}
            </Text>

            <Text style={styles.slotText}>
              Slot Number: {selectedClassSlot.slot_no}
            </Text>

            <Text style={styles.studentCount}>
              {registeredStudents.length} registered student(s)
            </Text>
          </View>

          {loadingStudents ? (
            <ActivityIndicator
              size="large"
              color="#2563eb"
              style={styles.loader}
            />
          ) : (
            <FlatList
              data={registeredStudents}
              keyExtractor={item => item.registration_number}
              renderItem={renderStudent}
              contentContainerStyle={styles.listContent}
              showsVerticalScrollIndicator={false}
              ListEmptyComponent={
                <Text style={styles.emptyText}>
                  No registered students found.
                </Text>
              }
            />
          )}
        </>
      )}
    </View>
  );
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    padding: 20,
  },

  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#111827',
  },

  subtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 18,
    marginTop: 8,
  },

  searchInput: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 10,
    padding: 13,
    fontSize: 15,
    marginBottom: 16,
    color: '#111827',
  },

  listContent: {
    paddingBottom: 30,
    flexGrow: 1,
  },

  courseCard: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 10,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },

  courseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  courseName: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#111827',
    flex: 1,
  },

  courseId: {
    fontSize: 14,
    color: '#2563eb',
    marginTop: 6,
  },

  studentCount: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 8,
  },

  arrow: {
    fontSize: 26,
    color: '#6b7280',
  },

  backButton: {
    marginBottom: 18,
    alignSelf: 'flex-start',
  },

  backText: {
    color: '#2563eb',
    fontSize: 16,
    fontWeight: '600',
  },

  classCard: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 10,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },

  classTitle: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#111827',
  },

  slotText: {
    fontSize: 14,
    color: '#2563eb',
    marginTop: 8,
  },

  selectedClassCard: {
    backgroundColor: '#fff',
    padding: 14,
    borderRadius: 10,
    marginTop: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },

  studentCard: {
    backgroundColor: '#fff',
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },

  studentPhoto: {
    width: 52,
    height: 52,
    borderRadius: 26,
    marginRight: 14,
  },

  photoPlaceholder: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#dbeafe',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 14,
  },

  photoPlaceholderText: {
    fontSize: 21,
    fontWeight: 'bold',
    color: '#2563eb',
  },

  studentInfo: {
    flex: 1,
    minWidth: 0,
  },

  studentName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#111827',
  },

  registrationNumber: {
    fontSize: 13,
    color: '#2563eb',
    marginTop: 4,
  },

  studentSlot: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },

  deleteButton: {
    backgroundColor: '#dc2626',
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderRadius: 7,
    marginLeft: 8,
    minWidth: 65,
    alignItems: 'center',
    justifyContent: 'center',
  },

  disabledDeleteButton: {
    opacity: 0.6,
  },

  deleteButtonText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: 'bold',
  },

  emptyText: {
    textAlign: 'center',
    color: '#6b7280',
    marginTop: 30,
  },

  loader: {
    marginTop: 30,
  },
});
