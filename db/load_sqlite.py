import sqlite3

# Connect to SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect('sqlite_data.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create the 'products' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) NOT NULL
)
''')

# Create the 'staff' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS staff (
    staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL
)
''')

# Create the 'orders' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    staff_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    FOREIGN KEY (staff_id) REFERENCES staff (staff_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
)
''')

# Insert data into the 'products' table
cursor.executemany('''
INSERT INTO products (product_name, price) VALUES (?, ?)
''', [
    ('Laptop', 799.99),
    ('Keyboard', 129.99),
    ('Mouse', 29.99)
])

# Insert data into the 'staff' table
cursor.executemany('''
INSERT INTO staff (first_name, last_name) VALUES (?, ?)
''', [
    ('Alice', 'Smith'),
    ('Bob', 'Johnson'),
    ('Charlie', 'Williams')
])

# Insert data into the 'orders' table
cursor.executemany('''
INSERT INTO orders (customer_name, staff_id, product_id) VALUES (?, ?, ?)
''', [
    ('David Lee', 1, 1),
    ('Emily Chen', 2, 2),
    ('Frank Brown', 1, 3)
])

# Commit the changes to the database
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("Tables created successfully.")
