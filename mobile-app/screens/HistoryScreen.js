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
import * as Sharing from 'expo-sharing';

import api, { getApiErrorMessage } from '../api';

export default function HistoryScreen() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exportingId, setExportingId] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(null);

  // ---------------------------------------------------------
  // Load attendance sessions
  // ---------------------------------------------------------
  const loadHistory = useCallback(async () => {
    setLoading(true);

    try {
      const res = await api.get('/attendance/sessions');

      setSessions(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      console.log('HISTORY ERROR:', err);
      console.log('HISTORY MESSAGE:', err.message);
      console.log('HISTORY RESPONSE:', err.response?.data);

      Alert.alert(
        'Error',
        getApiErrorMessage(err, 'Failed to load attendance sessions')
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
  // View attendance details
  // ---------------------------------------------------------
  const viewDetails = async (session) => {
    setDetailsLoading(session.id);

    try {
      const res = await api.get(
        `/attendance/history/${session.id}`
      );

      const data = res.data;

      const records = Array.isArray(data.records)
        ? data.records
        : [];

      const present = records.filter(
        (record) => record.status === 'present'
      ).length;

      const absent = records.filter(
        (record) => record.status === 'absent'
      ).length;

      const recordText = records
        .map((record) => {
          const registrationNumber =
            record.registration_number ||
            record.student?.registration_number ||
            'Unknown';

          const name =
            record.student_name ||
            record.name ||
            record.student?.name ||
            'Unknown';

          const similarity =
            record.similarity_score != null
              ? ` (${(record.similarity_score * 100).toFixed(1)}%)`
              : '';

          return `${registrationNumber} - ${name}: ${record.status}${similarity}`;
        })
        .join('\n');

      Alert.alert(
        `Session #${session.id}`,
        `Present: ${present}\nAbsent: ${absent}\n\n${recordText}`,
        [{ text: 'Close' }]
      );
    } catch (err) {
      console.log('DETAILS ERROR:', err);

      Alert.alert(
        'Error',
        getApiErrorMessage(
          err,
          'Failed to load attendance details'
        )
      );
    } finally {
      setDetailsLoading(null);
    }
  };

  // ---------------------------------------------------------
  // Export Excel
  // ---------------------------------------------------------
  const exportExcel = async (session) => {
    const sessionId = session.id;

    setExportingId(sessionId);

    try {
      const fileUri =
        `${FileSystem.documentDirectory}` +
        `attendance_${sessionId}.xlsx`;

      const downloadUrl =
        `${api.defaults.baseURL}/attendance/report/${sessionId}`;

      console.log('Downloading Excel:', downloadUrl);

      const result = await FileSystem.downloadAsync(
        downloadUrl,
        fileUri
      );

      if (result.status !== 200) {
        throw new Error(
          `Download failed with status ${result.status}`
        );
      }

      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(result.uri, {
          mimeType:
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          dialogTitle:
            `Session ${sessionId} attendance`,
        });
      } else {
        Alert.alert(
          'Downloaded',
          `Excel file saved to:\n${result.uri}`
        );
      }
    } catch (err) {
      console.log('EXPORT ERROR:', err);
      console.log('EXPORT RESPONSE:', err.response?.data);

      Alert.alert(
        'Export failed',
        getApiErrorMessage(
          err,
          'Could not download attendance report'
        )
      );
    } finally {
      setExportingId(null);
    }
  };

  // ---------------------------------------------------------
  // Render one session
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
          {item.class_name || 'Class'}
        </Text>

        {item.section ? (
          <Text style={styles.section}>
            Section {item.section}
          </Text>
        ) : null}
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