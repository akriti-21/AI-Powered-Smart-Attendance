import * as SQLite from 'expo-sqlite';

let database = null;

export async function initDatabase() {
  if (database) return database;

  database = await SQLite.openDatabaseAsync('attendance.db');

  await database.execAsync(`
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS students (
      registration_number TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      department TEXT,
      year INTEGER,
      section TEXT,
      photo_uri TEXT,
      embedding TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS attendance_sessions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      class_name TEXT NOT NULL,
      section TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      total_count INTEGER DEFAULT 0,
      present_count INTEGER DEFAULT 0,
      absent_count INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS attendance_records (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER NOT NULL,
      registration_number TEXT,
      student_name TEXT,
      status TEXT NOT NULL,
      similarity_score REAL,
      FOREIGN KEY (session_id)
        REFERENCES attendance_sessions(id)
        ON DELETE CASCADE
    );
  `);

  return database;
}

export async function getDatabase() {
  return database || await initDatabase();
}
