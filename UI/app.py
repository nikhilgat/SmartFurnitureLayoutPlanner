import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root) 

from flask import Flask, render_template, jsonify, request
from database.database import LayoutDatabase
from database.model_manager import get_model_manager
import time

app = Flask(__name__)

db = LayoutDatabase()

model_manager = get_model_manager()

print("\n" + "="*60)
print("STARTING FLASK SERVER")
print("Initiating background model loading...")
print("="*60 + "\n")
model_manager.start_loading()

# ==================== WEB ROUTES ====================

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

# ==================== MODEL STATUS API ====================

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Get current model loading status"""
    status = model_manager.get_loading_status()
    return jsonify(status)

@app.route('/api/queue-status', methods=['GET'])
def queue_status():
    """Get current optimization queue status"""
    status = model_manager.get_queue_status()
    return jsonify(status)

# ==================== OPTIMIZATION API ====================

@app.route('/api/optimize', methods=['POST'])
def optimize_layout():
    """
    Submit a layout for optimization
    
    Expected JSON body:
    {
        "layout": { ... layout data ... },
        "notes": "Optional notes about this layout"
    }
    
    Returns:
    {
        "success": true/false,
        "original_layout_id": <id>,
        "optimized_layout_id": <id>,
        "job_id": "<job_id>",
        "message": "..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'layout' not in data:
            return jsonify({
                'success': False,
                'error': 'No layout data provided'
            }), 400
        
        layout_data = data['layout']
        notes = data.get('notes', None)
        
        if not model_manager.is_loaded:
            return jsonify({
                'success': False,
                'error': 'Model is still loading. Please wait.'
            }), 503
        
        original_layout_id = db.save_original_layout(layout_data, notes)
        print(f"[API] Saved original layout with ID: {original_layout_id}")
        
        optimized_layout_id = db.save_optimized_layout(
            original_layout_id=original_layout_id,
            layout_data=None,
            status='pending'
        )
        print(f"[API] Created optimized layout placeholder with ID: {optimized_layout_id}")
        
        job_id = model_manager.submit_optimization(layout_data, original_layout_id)
        print(f"[API] Submitted optimization job: {job_id}")
        

        return jsonify({
            'success': True,
            'original_layout_id': original_layout_id,
            'optimized_layout_id': optimized_layout_id,
            'job_id': job_id,
            'message': 'Layout submitted for optimization'
        })
        
    except Exception as e:
        print(f"[API ERROR] optimize_layout: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/optimization-status/<job_id>', methods=['GET'])
def optimization_status(job_id):
    """
    Get the status of an optimization job
    
    Returns:
    {
        "job_id": "<job_id>",
        "status": "queued|processing|completed|failed",
        "progress": 0-100,
        "output_layout": { ... } (if completed),
        "violations_count": <number> (if completed),
        "error": "..." (if failed)
    }
    """
    try:
        status = model_manager.get_optimization_status(job_id)
        
        if status['status'] == 'completed' and 'output_layout' in status:
            original_layout_id = status['original_layout_id']
            optimized_layouts = db.get_optimized_by_original(original_layout_id)
            
            for opt in optimized_layouts:
                if opt['status'] == 'pending':
                    db.update_optimization_status(
                        optimized_id=opt['id'],
                        status='completed',
                        layout_data=status['output_layout'],
                        optimization_time=status['optimization_time'],
                        violations_count=status['violations_count']
                    )
                    print(f"[API] Updated database for optimized layout ID: {opt['id']}")
                    status['optimized_layout_id'] = opt['id']
                    break
        
        elif status['status'] == 'failed':
            original_layout_id = status['original_layout_id']
            optimized_layouts = db.get_optimized_by_original(original_layout_id)
            
            for opt in optimized_layouts:
                if opt['status'] == 'pending':
                    db.update_optimization_status(
                        optimized_id=opt['id'],
                        status='failed',
                        error_message=status.get('error', 'Unknown error')
                    )
                    print(f"[API] Marked optimized layout ID {opt['id']} as failed")
                    status['optimized_layout_id'] = opt['id']
                    break
        
        return jsonify(status)
        
    except Exception as e:
        print(f"[API ERROR] optimization_status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/get-optimized/<int:optimized_id>', methods=['GET'])
def get_optimized_layout(optimized_id):
    """
    Retrieve an optimized layout by ID
    
    Returns:
    {
        "success": true/false,
        "layout": { ... optimized layout data ... },
        "metadata": { ... }
    }
    """
    try:
        optimized = db.get_optimized_layout(optimized_id)
        
        if not optimized:
            return jsonify({
                'success': False,
                'error': 'Optimized layout not found'
            }), 404
        
        return jsonify({
            'success': True,
            'layout': optimized['layout_data'],
            'metadata': {
                'id': optimized['id'],
                'original_layout_id': optimized['original_layout_id'],
                'created_at': optimized['created_at'],
                'optimization_time_seconds': optimized['optimization_time_seconds'],
                'violations_count': optimized['violations_count'],
                'status': optimized['status']
            }
        })
        
    except Exception as e:
        print(f"[API ERROR] get_optimized_layout: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== LAYOUT MANAGEMENT API ====================

@app.route('/api/layouts/original', methods=['GET'])
def list_original_layouts():
    """
    Get list of all original layouts
    
    Query params:
    - limit: max number to return (default 50)
    - offset: pagination offset (default 0)
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        layouts = db.get_all_original_layouts(limit, offset)
        
        return jsonify({
            'success': True,
            'layouts': layouts,
            'count': len(layouts)
        })
        
    except Exception as e:
        print(f"[API ERROR] list_original_layouts: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/layouts/original/<int:layout_id>', methods=['GET'])
def get_original_layout(layout_id):
    """Get a specific original layout by ID"""
    try:
        layout = db.get_original_layout(layout_id)
        
        if not layout:
            return jsonify({
                'success': False,
                'error': 'Layout not found'
            }), 404
        
        return jsonify({
            'success': True,
            'layout': layout
        })
        
    except Exception as e:
        print(f"[API ERROR] get_original_layout: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/layouts/pair/<int:original_layout_id>', methods=['GET'])
def get_layout_pair(original_layout_id):
    """
    Get both original and optimized layout together
    
    Returns:
    {
        "success": true,
        "original": { ... },
        "optimized": { ... }
    }
    """
    try:
        pair = db.get_layout_pair(original_layout_id)
        
        if not pair['original']:
            return jsonify({
                'success': False,
                'error': 'Original layout not found'
            }), 404
        
        return jsonify({
            'success': True,
            'original': pair['original'],
            'optimized': pair['optimized']
        })
        
    except Exception as e:
        print(f"[API ERROR] get_layout_pair: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/layouts/<int:original_layout_id>', methods=['DELETE'])
def delete_layout_pair(original_layout_id):
    """Delete an original layout and all its optimized versions"""
    try:
        db.delete_layout_pair(original_layout_id)
        
        return jsonify({
            'success': True,
            'message': 'Layout pair deleted successfully'
        })
        
    except Exception as e:
        print(f"[API ERROR] delete_layout_pair: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== STATISTICS API ====================

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get database and system statistics"""
    try:
        db_stats = db.get_statistics()
        model_status = model_manager.get_loading_status()
        queue_status = model_manager.get_queue_status()
        
        return jsonify({
            'success': True,
            'database': db_stats,
            'model': model_status,
            'queue': queue_status
        })
        
    except Exception as e:
        print(f"[API ERROR] get_statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== UTILITY ROUTES ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_manager.is_loaded,
        'model_loading': model_manager.is_loading,
        'timestamp': time.time()
    })

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Flask server starting...")
    print("Model loading in background...")
    print("Access the app at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, threaded=False)