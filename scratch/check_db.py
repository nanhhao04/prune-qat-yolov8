import sqlite3
import os

db_path = "dataset/sherbrooke_annotations/sherbrooke_gt.sqlite"
if not os.path.exists(db_path):
    print("DB not found")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        info = cursor.fetchall()
        print(f"\nTable {table_name} schema:")
        for col in info:
            print(col)
        
        # Sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()
        print(f"Sample data from {table_name}:")
        for row in rows:
            print(row)
    conn.close()
