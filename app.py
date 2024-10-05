from flask import Flask, render_template, request
import face_recognition
import cv2
import numpy as np
import os
from PIL import Image, ExifTags

def load_image_file_with_orientation(file):
    img = Image.open(file)
    print(f"Original image mode: {img.mode}")
    try:
        # Get the EXIF orientation tag
        exif = img._getexif()
        if exif is not None:
            exif = dict(exif.items())
            orientation = exif.get(274)  # 274 is the EXIF tag code for Orientation
            print(f"Image EXIF orientation: {orientation}")
            # Adjust image orientation based on EXIF data
            if orientation == 3:
                # print("Rotating image 180 degrees")
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                # print("Rotating image -90 degrees (90 degrees CW)")
                img = img.rotate(-90, expand=True)
            elif orientation == 8:
                # print("Rotating image 90 degrees (90 degrees CCW)")
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"Error processing {file}: {e}")
    finally:
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"Converted image mode: {img.mode}")

    return np.array(img)

# Initialize the Flask app
app = Flask(__name__)

# Create arrays for known face encodings and names
known_face_encodings = []
known_face_names = []

# Load images and learn how to recognize them
dataset_dir = 'dataset'


for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = load_image_file_with_orientation(image_path)
        try:
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(person_name)
        except IndexError:
            print(f"No face found in {image_path}. Skipping.")
        except RuntimeError as e:
            print(f"Unsupported image type in {image_path}. Skipping. Error: {e}")


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        # Save the uploaded image
        image_path = os.path.join('static', 'uploaded_image.jpg')
        if not os.path.exists('static'):
            os.makedirs('static')
        file.save(image_path)

        # Recognize faces in the image
        recognized_names = recognize_faces(image_path)
        print(recognized_names)

        return render_template('result.html', names=recognized_names)
    return render_template('upload.html')


def recognize_faces(image_path):
    # Load the uploaded image
    unknown_image = load_image_file_with_orientation(image_path)

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    recognized_names = []

    # Convert image to BGR color for OpenCV
    image_cv = cv2.imread(image_path)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the closest match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if face_distances.size > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        recognized_names.append(name)

        # Draw a rectangle and label around the face
        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_cv, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save the result image
    result_image_path = os.path.join('static', 'result.jpg')
    cv2.imwrite(result_image_path, image_cv)

    return recognized_names

if __name__ == '__main__':
    app.run(debug=True)