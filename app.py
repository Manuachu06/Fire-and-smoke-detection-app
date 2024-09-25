from flask import Flask, request, jsonify, redirect, render_template, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import cv2
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'output/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a random secret key

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'register'  # Redirect to register if not logged in

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt')

# Ensure output folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
@login_required
def index():
    return render_template('index.html')

# User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user is already registered
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect('/register')

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect('/login')
    return render_template('register.html')

# User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect('/')
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

# User Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process based on file type
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        result_img = process_image(filepath)
        return render_template('index.html', result_img=result_img, is_video=False)
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        result_video = process_video(filepath)
        return render_template('index.html', result_video=result_video, is_video=True)
    else:
        return redirect(request.url)

def process_image(filepath):
    results = model(filepath)
    result_img = results.render()[0]
    result_img_path = os.path.join(app.config['OUTPUT_FOLDER'], f'result_{os.path.basename(filepath)}')
    cv2.imwrite(result_img_path, result_img)
    return f'result_{os.path.basename(filepath)}'

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f'processed_{os.path.basename(filepath)}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        result_img = results.render()[0]
        out.write(result_img)

    cap.release()
    out.release()
    return f'processed_{os.path.basename(filepath)}'

@app.route('/output/<path:filename>')
@login_required
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    # Create the database tables within the application context
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
