from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sqlite3

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key


# =======================
# Database setup
# =======================
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


# Initialize database
init_db()


# =======================
# Model loading
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../webapp
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'monkeypox_mobilenetv3_final.h5'))

print("Looking for model at:", MODEL_PATH)

model = None
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# =======================
# Login required decorator
# =======================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


# =======================
# Routes
# =======================

@app.route('/')
def home():
    # You can change this later if you want a separate public page
    return render_template('home.html')


@app.route('/home')
@login_required
def user_home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('detection'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, generate_password_hash(password))
            )
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'danger')
        finally:
            conn.close()

    return render_template('signup.html')


@app.route('/detection', methods=['GET', 'POST'])
@login_required
def detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No image selected', 'danger')
            return redirect(request.url)

        # Save file
        uploads_dir = os.path.join('static', 'uploads', str(session['user_id']))
        os.makedirs(uploads_dir, exist_ok=True)
        filepath = os.path.join(uploads_dir, file.filename)
        file.save(filepath)

        # Make prediction
        try:
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            if model:
                prediction = model.predict(img_array, verbose=0)[0][0]

                print("Raw prediction:", prediction)

                # In your model: 0 → Monkeypox, 1 → Others
                if prediction < 0.5:
                    label = "Monkeypox"
                    confidence = round((1 - prediction) * 100, 2)
                    return redirect(url_for('result_monkeypox', confidence=confidence))
                else:
                    label = "Non-monkeypox"
                    confidence = round(prediction * 100, 2)
                    return redirect(url_for('result_healthy'))
            else:
                flash('Model not loaded. Please try again later.', 'danger')
                return redirect(url_for('detection'))

        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(url_for('detection'))

    return render_template('detection.html')


@app.route('/result/monkeypox')
@login_required
def result_monkeypox():
    confidence = float(request.args.get('confidence', 0.7))
    return render_template('result_monkeypox.html', confidence=confidence)


@app.route('/result/healthy')
@login_required
def result_healthy():
    return render_template('result_healthy.html')


@app.route('/precautions')
@login_required
def precautions():
    return render_template('precautions.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
sssss