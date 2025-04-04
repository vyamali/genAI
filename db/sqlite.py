import sqlite3

class SQLiteDatabase:
    def __init__(self, db_file: str):
        """Initialize the SQLiteDatabase object with a given database file."""
        self.db_file = db_file
        self.db_conn = sqlite3.connect(db_file, check_same_thread=False)
        print(f" - DB CONNECTED: {db_file}")

    def list_tables(self) -> list[str]:
        """Retrieve the names of all tables in the database."""
        print(' - DB CALL: list_tables()')
        cursor = self.db_conn.cursor()

        # Fetch the table names.
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        tables = cursor.fetchall()
        return [t[0] for t in tables]

    def describe_table(self, table_name: str) -> list[tuple[str, str]]:
        """Look up the table schema.

        Returns:
          List of columns, where each entry is a tuple of (column, type).
        """
        print(f' - DB CALL: describe_table({table_name})')

        cursor = self.db_conn.cursor()

        cursor.execute(f"PRAGMA table_info({table_name});")

        schema = cursor.fetchall()
        # [column index, column name, column type, ...]
        return [(col[1], col[2]) for col in schema]

    def execute_query(self, sql: str) -> list[list[str]]:
        """Execute an SQL statement, returning the results."""
        print(f' - DB CALL: execute_query({sql})')

        cursor = self.db_conn.cursor()

        cursor.execute(sql)
        return cursor.fetchall()

    def close(self):
        """Close the database connection."""
        self.db_conn.close()
        print(f" - DB CONNECTION CLOSED")
