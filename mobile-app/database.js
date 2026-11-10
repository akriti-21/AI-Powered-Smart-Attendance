
import * as SQLite from 'expo-sqlite';

let database = null;
let initializationPromise = null;

// ---------------------------------------------------------
// Add a column only if it does not already exist
// ---------------------------------------------------------

async function addColumnIfMissing(
  db,
  tableName,
  columnName,
  columnType
) {
  const columns = await db.getAllAsync(
    `PRAGMA table_info(${tableName})`
  );

  const exists = columns.some(
    (column) => column.name === columnName
  );

  if (exists) {
    console.log(`${tableName}.${columnName} already exists`);
    return;
  }

  try {
    await db.execAsync(
      `ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${columnType}`
    );

    console.log(`Added ${tableName}.${columnName}`);

  } catch (error) {
    if (
      !String(error.message)
        .toLowerCase()
        .includes('duplicate column')
    ) {
      throw error;
    }

    console.log(`${tableName}.${columnName} already exists`);
  }
}

// ---------------------------------------------------------
// Initialize local SQLite database
// ---------------------------------------------------------

export async function initDatabase() {

  if (database) {
    return database;
  }

  if (initializationPromise) {
    return initializationPromise;
  }

  initializationPromise = (async () => {

    const db = await SQLite.openDatabaseAsync(
      'attendance.db'
    );

    await db.execAsync(`
      PRAGMA foreign_keys = ON;

      CREATE TABLE IF NOT EXISTS students (
        registration_number TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        year INTEGER,
        section TEXT,
        course_name TEXT,
        course_id TEXT,
        class_id TEXT,
        slot_no TEXT,
        photo_uri TEXT,
        embedding TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS attendance_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT NOT NULL,
        section TEXT,
        course_name TEXT,
        course_id TEXT,
        class_id TEXT,
        slot_no TEXT,
        image_uri TEXT,
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

    // -----------------------------------------------------
    // Migrate existing students table safely
    // -----------------------------------------------------

    const studentColumns = [
      ['course_name', 'TEXT'],
      ['course_id', 'TEXT'],
      ['class_id', 'TEXT'],
      ['slot_no', 'TEXT'],
      ['photo_uri', 'TEXT'],
      ['embedding', 'TEXT'],
    ];

    for (const [name, type] of studentColumns) {
      await addColumnIfMissing(
        db,
        'students',
        name,
        type
      );
    }

    // -----------------------------------------------------
    // Migrate existing attendance sessions safely
    // -----------------------------------------------------

    const sessionColumns = [
      ['course_name', 'TEXT'],
      ['course_id', 'TEXT'],
      ['class_id', 'TEXT'],
      ['slot_no', 'TEXT'],
      ['image_uri', 'TEXT'],
      ['total_count', 'INTEGER DEFAULT 0'],
      ['present_count', 'INTEGER DEFAULT 0'],
      ['absent_count', 'INTEGER DEFAULT 0'],
    ];

    for (const [name, type] of sessionColumns) {
      await addColumnIfMissing(
        db,
        'attendance_sessions',
        name,
        type
      );
    }

    // -----------------------------------------------------
    // Check existing student embeddings
    // -----------------------------------------------------

    const embeddingStatus = await db.getAllAsync(`
      SELECT
        registration_number,
        name,
        CASE
          WHEN embedding IS NULL OR embedding = ''
          THEN 'MISSING'
          ELSE 'SAVED'
        END AS embedding_status
      FROM students
    `);

    console.log(
      'STUDENT EMBEDDING STATUS:',
      embeddingStatus
    );

    database = db;

    console.log('Database initialized successfully');

    return db;

  })();

  try {

    return await initializationPromise;

  } catch (error) {

    initializationPromise = null;

    console.error(
      'Database initialization failed:',
      error
    );

    throw error;
  }
}

// ---------------------------------------------------------
// Get initialized database
// ---------------------------------------------------------

export async function getDatabase() {

  if (database) {
    return database;
  }

  return await initDatabase();
}
