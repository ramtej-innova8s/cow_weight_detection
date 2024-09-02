import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from weight_detection import process_video  # Import the process_video function

app = Flask(__name__)

# Ensure the uploads and outputs directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        # Process video
        output_path = os.path.join('outputs', f"processed_{file.filename}")
        processed_video = process_video(filepath, output_path)
        
        return redirect(url_for('download', filename=os.path.basename(processed_video)))

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    app.run(debug=True)