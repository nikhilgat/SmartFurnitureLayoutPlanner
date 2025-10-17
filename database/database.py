import sqlite3
import json
from datetime import datetime
import os
import threading

class LayoutDatabase:
    def __init__(self, db_path='database/layouts.db'):
        """Initialize database connection and create tables if they don't exist"""
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._lock = threading.Lock()  # Thread safety
        self.init_db()
    
    def get_connection(self):
        """Get a database connection with timeout"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        return conn
    
    def init_db(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Original layouts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS original_layouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layout_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                room_width INTEGER,
                room_height INTEGER,
                furniture_count INTEGER,
                notes TEXT
            )
        ''')
        
        # Optimized layouts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimized_layouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_layout_id INTEGER NOT NULL,
                layout_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                optimization_time_seconds REAL,
                violations_count INTEGER,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                FOREIGN KEY (original_layout_id) REFERENCES original_layouts (id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_original_created 
            ON original_layouts(created_at)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_optimized_original 
            ON optimized_layouts(original_layout_id)
        ''')
        
        conn.commit()
        conn.close()
    
    # ==================== ORIGINAL LAYOUTS ====================
    
    def save_original_layout(self, layout_data, notes=None):
        """
        Save an original layout to the database
        
        Args:
            layout_data (dict): The layout JSON data
            notes (str, optional): Additional notes about the layout
            
        Returns:
            int: The ID of the saved layout
        """
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Extract metadata
            room_width = layout_data.get('room', {}).get('width')
            room_height = layout_data.get('room', {}).get('height')
            furniture_count = len(layout_data.get('furniture', []))
            
            cursor.execute('''
                INSERT INTO original_layouts 
                (layout_data, room_width, room_height, furniture_count, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                json.dumps(layout_data),
                room_width,
                room_height,
                furniture_count,
                notes
            ))
            
            layout_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return layout_id
    
    def get_original_layout(self, layout_id):
        """
        Retrieve an original layout by ID
        
        Args:
            layout_id (int): The layout ID
            
        Returns:
            dict: Layout data with metadata, or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM original_layouts WHERE id = ?
        ''', (layout_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'layout_data': json.loads(row['layout_data']),
                'created_at': row['created_at'],
                'room_width': row['room_width'],
                'room_height': row['room_height'],
                'furniture_count': row['furniture_count'],
                'notes': row['notes']
            }
        return None
    
    def get_all_original_layouts(self, limit=50, offset=0):
        """
        Get all original layouts with pagination
        
        Args:
            limit (int): Maximum number of layouts to return
            offset (int): Number of layouts to skip
            
        Returns:
            list: List of layout metadata (without full layout_data)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, created_at, room_width, room_height, 
                   furniture_count, notes
            FROM original_layouts
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ==================== OPTIMIZED LAYOUTS ====================
    
    def save_optimized_layout(self, original_layout_id, layout_data, 
                             optimization_time=None, violations_count=0, 
                             status='completed', error_message=None):
        """
        Save an optimized layout to the database
        
        Args:
            original_layout_id (int): ID of the original layout
            layout_data (dict): The optimized layout JSON data
            optimization_time (float, optional): Time taken to optimize in seconds
            violations_count (int): Number of violations found
            status (str): Status of optimization (pending, completed, failed)
            error_message (str, optional): Error message if optimization failed
            
        Returns:
            int: The ID of the saved optimized layout
        """
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO optimized_layouts 
                (original_layout_id, layout_data, optimization_time_seconds, 
                 violations_count, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                original_layout_id,
                json.dumps(layout_data) if layout_data else None,
                optimization_time,
                violations_count,
                status,
                error_message
            ))
            
            optimized_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return optimized_id
    
    def get_optimized_layout(self, optimized_id):
        """
        Retrieve an optimized layout by ID
        
        Args:
            optimized_id (int): The optimized layout ID
            
        Returns:
            dict: Optimized layout data with metadata, or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM optimized_layouts WHERE id = ?
        ''', (optimized_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'original_layout_id': row['original_layout_id'],
                'layout_data': json.loads(row['layout_data']) if row['layout_data'] else None,
                'created_at': row['created_at'],
                'optimization_time_seconds': row['optimization_time_seconds'],
                'violations_count': row['violations_count'],
                'status': row['status'],
                'error_message': row['error_message']
            }
        return None
    
    def get_optimized_by_original(self, original_layout_id):
        """
        Get all optimized layouts for a given original layout
        
        Args:
            original_layout_id (int): The original layout ID
            
        Returns:
            list: List of optimized layouts
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM optimized_layouts 
            WHERE original_layout_id = ?
            ORDER BY created_at DESC
        ''', (original_layout_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'id': row['id'],
                'original_layout_id': row['original_layout_id'],
                'layout_data': json.loads(row['layout_data']) if row['layout_data'] else None,
                'created_at': row['created_at'],
                'optimization_time_seconds': row['optimization_time_seconds'],
                'violations_count': row['violations_count'],
                'status': row['status'],
                'error_message': row['error_message']
            })
        
        return results
    
    def update_optimization_status(self, optimized_id, status, 
                                   error_message=None, layout_data=None,
                                   optimization_time=None, violations_count=None):
        """
        Update the status of an optimization
        
        Args:
            optimized_id (int): The optimized layout ID
            status (str): New status (pending, completed, failed)
            error_message (str, optional): Error message if failed
            layout_data (dict, optional): Updated layout data
            optimization_time (float, optional): Time taken
            violations_count (int, optional): Number of violations
        """
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build dynamic UPDATE query
            updates = ['status = ?']
            params = [status]
            
            if error_message is not None:
                updates.append('error_message = ?')
                params.append(error_message)
            
            if layout_data is not None:
                updates.append('layout_data = ?')
                params.append(json.dumps(layout_data))
            
            if optimization_time is not None:
                updates.append('optimization_time_seconds = ?')
                params.append(optimization_time)
            
            if violations_count is not None:
                updates.append('violations_count = ?')
                params.append(violations_count)
            
            params.append(optimized_id)
            
            query = f'''
                UPDATE optimized_layouts 
                SET {', '.join(updates)}
                WHERE id = ?
            '''
            
            cursor.execute(query, params)
            conn.commit()
            conn.close()
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def get_layout_pair(self, original_layout_id):
        """
        Get both original and its most recent optimized layout
        
        Args:
            original_layout_id (int): The original layout ID
            
        Returns:
            dict: Contains 'original' and 'optimized' layout data
        """
        original = self.get_original_layout(original_layout_id)
        optimized_list = self.get_optimized_by_original(original_layout_id)
        
        return {
            'original': original,
            'optimized': optimized_list[0] if optimized_list else None
        }
    
    def delete_layout_pair(self, original_layout_id):
        """
        Delete an original layout and all its optimized versions
        
        Args:
            original_layout_id (int): The original layout ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Delete optimized layouts first (foreign key constraint)
        cursor.execute('''
            DELETE FROM optimized_layouts 
            WHERE original_layout_id = ?
        ''', (original_layout_id,))
        
        # Delete original layout
        cursor.execute('''
            DELETE FROM original_layouts 
            WHERE id = ?
        ''', (original_layout_id,))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self):
        """
        Get database statistics
        
        Returns:
            dict: Statistics about layouts
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Count original layouts
        cursor.execute('SELECT COUNT(*) as count FROM original_layouts')
        original_count = cursor.fetchone()['count']
        
        # Count optimized layouts
        cursor.execute('SELECT COUNT(*) as count FROM optimized_layouts')
        optimized_count = cursor.fetchone()['count']
        
        # Average optimization time
        cursor.execute('''
            SELECT AVG(optimization_time_seconds) as avg_time 
            FROM optimized_layouts 
            WHERE status = 'completed'
        ''')
        avg_time = cursor.fetchone()['avg_time']
        
        conn.close()
        
        return {
            'total_original_layouts': original_count,
            'total_optimized_layouts': optimized_count,
            'average_optimization_time': avg_time or 0
        }


# Example usage and testing
if __name__ == '__main__':
    # Initialize database
    db = LayoutDatabase()
    
    # Test data
    test_layout = {
        "room": {"width": 400, "height": 300},
        "furniture": [
            {"name": "Bed", "x": 50, "y": 50, "width": 160, "height": 200}
        ],
        "openings": []
    }
    
    # Save original layout
    original_id = db.save_original_layout(test_layout, notes="Test layout")
    print(f"Saved original layout with ID: {original_id}")
    
    # Save optimized layout
    optimized_id = db.save_optimized_layout(
        original_id, 
        test_layout, 
        optimization_time=2.5,
        violations_count=0,
        status='completed'
    )
    print(f"Saved optimized layout with ID: {optimized_id}")
    
    # Retrieve pair
    pair = db.get_layout_pair(original_id)
    print(f"\nRetrieved layout pair:")
    print(f"Original: {pair['original']['id']}")
    print(f"Optimized: {pair['optimized']['id']}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics:")
    print(f"Total original layouts: {stats['total_original_layouts']}")
    print(f"Total optimized layouts: {stats['total_optimized_layouts']}")
    print(f"Average optimization time: {stats['average_optimization_time']:.2f}s")