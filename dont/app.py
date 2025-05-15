from flask import Flask, render_template, request, redirect, url_for, session
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session management (replace with a secure key in production)

# Set paths
UPLOAD_FOLDER = 'data'
MODEL_PATH = 'modelss/signature_model1.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure required folders exist
os.makedirs(os.path.join(UPLOAD_FOLDER, 'genuine'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'forged'), exist_ok=True)
os.makedirs('modelss', exist_ok=True)

# Load or initialize the model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Simple user database (replace with a real database and hashed passwords in production)
users = {
    'admin': 'password123',  # Username: admin, Password: password123
    'ravi': 'securepass'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Middleware to check if user is logged in
def login_required(f):
    def wrap(*args, **kwargs):
        if 'username' not in session:
            print("Session does not contain 'username', redirecting to login")
            return redirect(url_for('login'))
        print(f"User {session['username']} is logged in")
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('signup.html', error='Username already exists')
        users[username] = password  # In production, hash the password
        print(f"New user registered: {username}")
        return render_template('signup.html', message='Registration successful! Please log in.')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            print(f"User {username} logged in successfully")
            return redirect(url_for('home'))
        print("Login failed: Invalid username or password")
        return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    print("Logging out user:", session.get('username'))
    session.clear()  # Clear all session data to ensure complete logout
    print("Session after logout:", session)
    return redirect(url_for('login', message='You have been logged out successfully'))

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/upload_and_train', methods=['POST'])
@login_required
def upload_and_train():
    if 'file' not in request.files or 'signature_type' not in request.form:
        return redirect(request.url)

    file = request.files['file']
    signature_type = request.form['signature_type']

    if file and allowed_file(file.filename):
        # Save the uploaded file in the respective folder
        label_folder = os.path.join(UPLOAD_FOLDER, signature_type)
        filepath = os.path.join(label_folder, file.filename)
        file.save(filepath)

        # Retrain the model
        train_model()

        return render_template('index.html', train_result="Model retrained successfully!")

    return redirect(request.url)

@app.route('/verify_signature', methods=['POST'])
@login_required
def verify_signature():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        result = "Forged Signature" if prediction[0][0] < 0.5 else "Genuine Signature"

        return render_template('index.html', result=result)

    return redirect(request.url)

def train_model():
    # Create data generators for training
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        UPLOAD_FOLDER,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Train the model
    model.fit(train_generator, epochs=50)

    # Save the updated model
    model.save(MODEL_PATH)

if __name__ == '__main__':
    app.run(debug=True)