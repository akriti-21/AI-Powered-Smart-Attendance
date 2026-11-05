import React, { useEffect, useState } from 'react';

import {
  View,
  Text,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';

import { StatusBar } from 'expo-status-bar';

import { NavigationContainer } from '@react-navigation/native';

import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import { Ionicons } from '@expo/vector-icons';

import RegisterScreen from './screens/RegisterScreen';

import AttendanceScreen from './screens/AttendanceScreen';

import HistoryScreen from './screens/HistoryScreen';

import { initDatabase } from './database';

const Tab = createBottomTabNavigator();

const TAB_ICONS = {
  Register: 'person-add',
  Attendance: 'camera',
  History: 'time',
};

export default function App() {
  const [dbReady, setDbReady] = useState(false);
  const [dbError, setDbError] = useState(null);

  useEffect(() => {
    initDatabase()
      .then(() => setDbReady(true))
      .catch((error) => {
        console.error('Database initialization failed:', error);
        setDbError(error.message);
      });
  }, []);

  if (dbError) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorTitle}>Database Error</Text>
        <Text>{dbError}</Text>
      </View>
    );
  }

  if (!dbReady) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.loadingText}>
          Preparing Attendance System...
        </Text>
      </View>
    );
  }

  return (
    <NavigationContainer>
      <StatusBar style="auto" />

      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerTitle: 'Attendance System',
          tabBarActiveTintColor: '#2563eb',
          tabBarInactiveTintColor: '#6b7280',
          tabBarIcon: ({ color, size }) => (
            <Ionicons
              name={TAB_ICONS[route.name]}
              size={size}
              color={color}
            />
          ),
        })}
      >
        <Tab.Screen name="Register" component={RegisterScreen} />
        <Tab.Screen name="Attendance" component={AttendanceScreen} />
        <Tab.Screen name="History" component={HistoryScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
  },
});
