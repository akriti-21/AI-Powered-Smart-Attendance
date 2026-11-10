
import React, { useCallback, useState } from 'react';

import {
  View,
  Text,
  TouchableOpacity,
  FlatList,
  Alert,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';

import { Ionicons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import * as IntentLauncher from 'expo-intent-launcher';

import { getDatabase } from '../database';

export default function HistoryScreen() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exportingId, setExportingId] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(null);

  // ---------------------------------------------------------
  // Load attendance sessions from SQLite
  // ---------------------------------------------------------

  const loadHistory = useCallback(async () => {
    setLoading(true);

    try {
      const db = await getDatabase();

      const result = await db.getAllAsync(`
        SELECT
          s.id,
          s.class_name,
          s.section,
          s.course_name,
          s.course_id,
          s.class_id,
          s.slot_no,
          s.created_at,
          COUNT(r.id) AS total,
          COALESCE(
            SUM(
              CASE
                WHEN LOWER(r.status) = 'present' THEN 1
                ELSE 0
              END
            ), 0
          ) AS present,
          COALESCE(
            SUM(
              CASE
                WHEN LOWER(r.status) = 'absent' THEN 1
                ELSE 0
              END
            ), 0
          ) AS absent
        FROM attendance_sessions s
        LEFT JOIN attendance_records r
          ON s.id = r.session_id
        GROUP BY s.id
        ORDER BY s.id DESC;
      `);

      const formattedSessions = result.map((item) => ({
        ...item,
        date: item.created_at,
        total: Number(item.total || 0),
        present: Number(item.present || 0),
        absent: Number(item.absent || 0),
      }));

      setSessions(formattedSessions);

      console.log('OFFLINE HISTORY LOADED:', formattedSessions);
    } catch (error) {
      console.error('OFFLINE HISTORY ERROR:', error);

      Alert.alert(
        'Error',
        'Could not load attendance history from the local database.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // ---------------------------------------------------------
  // Format date
  // ---------------------------------------------------------

  const formatDate = (dateString) => {
    if (!dateString) {
      return 'Unknown date';
    }

    const date = new Date(dateString);

    if (Number.isNaN(date.getTime())) {
      return String(dateString);
    }

    return date.toLocaleString();
  };

  // ---------------------------------------------------------
  // View attendance details from SQLite
  // ---------------------------------------------------------

  const viewDetails = async (session) => {
    setDetailsLoading(session.id);

    try {
      const db = await getDatabase();

      const records = await db.getAllAsync(
        `SELECT
           registration_number,
           student_name,
           status,
           similarity_score
         FROM attendance_records
         WHERE session_id = ?
         ORDER BY id ASC`,
        [session.id]
      );

      const present = records.filter(
        (record) =>
          String(record.status).toLowerCase() === 'present'
      ).length;

      const absent = records.filter(
        (record) =>
          String(record.status).toLowerCase() === 'absent'
      ).length;

      const recordText = records
        .map((record) => {
          const registrationNumber =
            record.registration_number || 'Unknown';

          const name =
            record.student_name || 'Unknown';

          const similarity =
            record.similarity_score != null
              ? ` (${(
                  Number(record.similarity_score) * 100
                ).toFixed(1)}%)`
              : '';

          return `${registrationNumber} - ${name}: ${record.status}${similarity}`;
        })
        .join('\n');

      Alert.alert(
        `Session #${session.id}`,
        `Course: ${session.course_name || 'N/A'}\n` +
          `Course ID: ${session.course_id || 'N/A'}\n` +
          `Class ID: ${
            session.class_id || session.class_name || 'N/A'
          }\n\n` +
          `Present: ${present}\n` +
          `Absent: ${absent}\n\n` +
          `${recordText || 'No attendance records found.'}`,
        [{ text: 'Close' }]
      );
    } catch (error) {
      console.error('OFFLINE DETAILS ERROR:', error);

      Alert.alert(
        'Error',
        'Could not load attendance details from the local database.'
      );
    } finally {
      setDetailsLoading(null);
    }
  };

  // ---------------------------------------------------------
  // Export attendance report locally as CSV
  // ---------------------------------------------------------

  const exportExcel = async (session) => {
    setExportingId(session.id);

    try {
      const db = await getDatabase();

      const records = await db.getAllAsync(
        `SELECT
           registration_number,
           student_name,
           status,
           similarity_score
         FROM attendance_records
         WHERE session_id = ?
         ORDER BY id ASC`,
        [session.id]
      );

      const escapeCSV = (value) => {
        const safeValue = String(value ?? '').replace(/"/g, '""');
        return `"${safeValue}"`;
      };

      const rows = [
        [
          'Registration Number',
          'Student Name',
          'Status',
          'Similarity Score',
        ],
        ...records.map((record) => [
          record.registration_number,
          record.student_name,
          record.status,
          record.similarity_score ?? '',
        ]),
      ];

      const csvContent = rows
        .map((row) => row.map(escapeCSV).join(','))
        .join('\n');

      const fileUri =
        `${FileSystem.documentDirectory}attendance_${session.id}.csv`;

      await FileSystem.writeAsStringAsync(
        fileUri,
        csvContent,
        {
          encoding: FileSystem.EncodingType.UTF8,
        }
      );

      console.log('OFFLINE REPORT CREATED:', fileUri);

      const contentUri =
        await FileSystem.getContentUriAsync(fileUri);

      await IntentLauncher.startActivityAsync(
        'android.intent.action.VIEW',
        {
          data: contentUri,
          flags: 1,
          type: 'text/csv',
        }
      );

      console.log('OFFLINE REPORT OPENED');
    } catch (error) {
      console.error('OFFLINE EXPORT ERROR:', error);

      Alert.alert(
        'Export failed',
        'Could not create or open the attendance report. Please check whether an app capable of opening CSV files is installed.'
      );
    } finally {
      setExportingId(null);
    }
  };

  // ---------------------------------------------------------
  // Render one attendance session
  // ---------------------------------------------------------

  const renderItem = ({ item }) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.titleContainer}>
          <Text style={styles.sessionTitle}>
            Session #{item.id}
          </Text>

          <Text style={styles.date}>
            {formatDate(item.date)}
          </Text>
        </View>

        <Ionicons
          name="calendar"
          size={26}
          color="#2563eb"
        />
      </View>

      <View style={styles.classInfo}>
        <Text style={styles.className}>
          {item.course_name || 'Course not available'}
        </Text>

        <Text style={styles.section}>
          Course ID: {item.course_id || 'N/A'}
        </Text>

        <Text style={styles.section}>
          Class ID: {item.class_id || item.class_name || 'N/A'}
        </Text>
      </View>

      <View style={styles.statsContainer}>
        <View style={styles.statBox}>
          <Text style={styles.statNumber}>
            {item.total}
          </Text>

          <Text style={styles.statLabel}>
            Total
          </Text>
        </View>

        <View style={styles.statBox}>
          <Text
            style={[
              styles.statNumber,
              styles.presentText,
            ]}
          >
            {item.present}
          </Text>

          <Text style={styles.statLabel}>
            Present
          </Text>
        </View>

        <View style={styles.statBox}>
          <Text
            style={[
              styles.statNumber,
              styles.absentText,
            ]}
          >
            {item.absent}
          </Text>

          <Text style={styles.statLabel}>
            Absent
          </Text>
        </View>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.detailsButton}
          onPress={() => viewDetails(item)}
          disabled={detailsLoading === item.id}
        >
          {detailsLoading === item.id ? (
            <ActivityIndicator
              size="small"
              color="#2563eb"
            />
          ) : (
            <>
              <Ionicons
                name="eye"
                size={18}
                color="#2563eb"
              />

              <Text style={styles.detailsText}>
                View Details
              </Text>
            </>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.exportButton}
          onPress={() => exportExcel(item)}
          disabled={exportingId !== null}
        >
          {exportingId === item.id ? (
            <ActivityIndicator
              size="small"
              color="#ffffff"
            />
          ) : (
            <>
              <Ionicons
                name="download"
                size={18}
                color="#ffffff"
              />

              <Text style={styles.exportText}>
                Excel
              </Text>
            </>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );

  // ---------------------------------------------------------
  // Screen
  // ---------------------------------------------------------

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.sectionTitle}>
          Attendance Sessions
        </Text>

        <TouchableOpacity
          onPress={loadHistory}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator
              size="small"
              color="#2563eb"
            />
          ) : (
            <Ionicons
              name="reload"
              size={25}
              color="#2563eb"
            />
          )}
        </TouchableOpacity>
      </View>

      {loading && sessions.length === 0 ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator
            size="large"
            color="#2563eb"
          />

          <Text style={styles.loadingText}>
            Loading attendance sessions...
          </Text>
        </View>
      ) : (
        <FlatList
          data={sessions}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderItem}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Ionicons
                name="calendar-outline"
                size={55}
                color="#9ca3af"
              />

              <Text style={styles.emptyTitle}>
                No attendance sessions yet
              </Text>

              <Text style={styles.emptyText}>
                Mark attendance first and the session
                will appear here.
              </Text>
            </View>
          }
          contentContainerStyle={[
            styles.listContent,
            sessions.length === 0 &&
              styles.emptyListContent,
          ]}
        />
      )}
    </View>
  );
}

// ---------------------------------------------------------
// Styles
// ---------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    padding: 16,
    paddingTop: 40,
  },

  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 18,
  },

  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111827',
  },

  loadingContainer: {
    alignItems: 'center',
    marginTop: 40,
  },

  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#6b7280',
  },

  card: {
    backgroundColor: '#ffffff',
    borderRadius: 14,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },

  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },

  titleContainer: {
    flex: 1,
  },

  sessionTitle: {
    fontSize: 19,
    fontWeight: 'bold',
    color: '#111827',
  },

  date: {
    marginTop: 4,
    fontSize: 13,
    color: '#6b7280',
  },

  classInfo: {
    marginTop: 12,
  },

  className: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2563eb',
  },

  section: {
    marginTop: 3,
    fontSize: 14,
    color: '#6b7280',
  },

  statsContainer: {
    flexDirection: 'row',
    marginTop: 16,
    gap: 8,
  },

  statBox: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: 'center',
  },

  statNumber: {
    fontSize: 21,
    fontWeight: 'bold',
    color: '#111827',
  },

  statLabel: {
    marginTop: 3,
    fontSize: 12,
    color: '#6b7280',
  },

  presentText: {
    color: '#16a34a',
  },

  absentText: {
    color: '#dc2626',
  },

  actions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 16,
  },

  detailsButton: {
    flex: 1,
    height: 44,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#2563eb',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
  },

  detailsText: {
    color: '#2563eb',
    fontWeight: '600',
  },

  exportButton: {
    flex: 1,
    height: 44,
    borderRadius: 9,
    backgroundColor: '#2563eb',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
  },

  exportText: {
    color: '#ffffff',
    fontWeight: '600',
  },

  emptyContainer: {
    alignItems: 'center',
    marginTop: 70,
    paddingHorizontal: 25,
  },

  emptyTitle: {
    marginTop: 15,
    fontSize: 18,
    fontWeight: '600',
    color: '#374151',
  },

  emptyText: {
    marginTop: 7,
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    lineHeight: 20,
  },

  listContent: {
    paddingBottom: 30,
  },

  emptyListContent: {
    flexGrow: 1,
  },
});
