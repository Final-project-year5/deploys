import face_recognition
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
import math

import imutils
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Adjust based on the pre-trained model you are using

mpl.use('Agg')

######################################for the the attendance post request
# # Load the trained model
# current_directory = os.path.dirname(os.path.abspath(__file__))
# model_filename = 'newmodel.h5'
# model_path = os.path.join(current_directory, model_filename)

# print("Current working directory:", os.getcwd())
# print("Constructed model path:", model_path)

# if not os.path.exists(model_path):
#     raise ValueError(f"Model file not found: {model_path}")

# try:
#     model = load_model(model_path)
#     print("Model loaded successfully")
#     model.summary()
# except Exception as e:
#     raise ValueError(f"Error loading the model: {e}")

# def preprocess_image(image):
#     try:
#         if image is not None:
#             print("Original image shape:", image.shape)  # Debugging statement
#             resized_image = cv2.resize(image, (224, 224))
#             print("Resized image shape:", resized_image.shape)  # Debugging statement
#             normalized_image = resized_image.astype('float32') / 255.0
#             processed_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
#             print("Processed image shape (after adding batch dimension):", processed_image.shape)  # Debugging statement
#             return processed_image
#         else:
#             raise ValueError("Input image is None")
#     except Exception as e:
#         print("Error preprocessing image:", e)
#         return None

# def perform_inference(model, processed_image):
#     try:
#         if processed_image is not None:
#             prediction = model.predict(processed_image)
#             return prediction
#         else:
#             raise ValueError("Processed image is None")
#     except Exception as e:
#         print("Error performing inference:", e)
#         return None

# def compare_images(input_features, database_features):
#     similarity_scores = [1 - cosine(input_features, db_features) for db_features in database_features]
#     threshold = 0.8
#     return max(similarity_scores) > threshold

# @csrf_exempt
# def predict(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         try:
#             image_file = request.FILES['image']
#             image = np.asarray(bytearray(image_file.read()), dtype="uint8")
#             image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#             if image is None:
#                 raise ValueError("Failed to decode image")

#             print("Decoded image shape:", image.shape)
#             processed_image = preprocess_image(image)
#             input_features = perform_inference(model, processed_image)
#             if input_features is None:
#                 raise ValueError("Failed to perform inference on input image")

#         except Exception as e:
#             print(f"Error reading or decoding image: {e}")
#             return JsonResponse({'error': f"Error reading or decoding image: {e}"}, status=500)

#         try:
#             database_images = FaceData.objects.all()
#             if database_images is not None and database_images.exists():
#                 database_features = []
#                 for db_image in database_images:
#                     db_image_path = db_image.face_encoding.path
#                     db_image_np = cv2.imread(db_image_path)
#                     if db_image_np is None:
#                         print(f"Failed to load database image from path: {db_image_path}")
#                         continue
#                     db_image_processed = preprocess_image(db_image_np)
#                     db_image_features = perform_inference(model, db_image_processed)
#                     if db_image_features is not None:
#                         database_features.append(db_image_features.flatten())

#                 if not database_features:
#                     return JsonResponse({'error': 'No valid database features found'}, status=404)

#                 match_found = compare_images(input_features.flatten(), database_features)
#                 if match_found:
#                     return JsonResponse({'message': 'Attendance recorded as present'}, status=200)
#                 else:
#                     return JsonResponse({'message': 'No matching record found'}, status=200)
#             else:
#                 return JsonResponse({'error': 'No database records found'}, status=404)
#         except Exception as e:
#             print("Error processing database:", e)
#             return JsonResponse({'error': 'Error processing database: No valid database features found'}, status=500)
# def check_image_match(new_image_path):
#     try:
#         absolute_image_path = os.path.abspath(new_image_path)
#         print(f"Absolute image path: {absolute_image_path}")

#         if not os.path.exists(absolute_image_path):
#             print(f"File does not exist: {absolute_image_path}")
#             parent_directory = os.path.dirname(absolute_image_path)
#             print(f"Contents of the directory {parent_directory}: {os.listdir(parent_directory)}")
#             return None

#         new_image_np = cv2.imread(absolute_image_path)
#         if new_image_np is None:
#             print(f"Failed to load image from path: {absolute_image_path}")
#             return None

#         print("Loaded new image shape:", new_image_np.shape)  # Debugging statement

#         processed_image = preprocess_image(new_image_np)
#         new_image_features = perform_inference(model, processed_image)

#         if new_image_features is not None:
#             database_images = FaceData.objects.all()
#             for db_image in database_images:
#                 db_image_path = db_image.face_encoding.path
#                 if not os.path.exists(db_image_path):
#                     print(f"Failed to load database image from path: {db_image_path}")
#                     continue
#                 db_image_np = cv2.imread(db_image_path)
#                 if db_image_np is None:
#                     print(f"Failed to load database image from path: {db_image_path}")
#                     continue
#                 print("Loaded database image shape:", db_image_np.shape)  # Debugging statement

#                 db_image_processed = preprocess_image(db_image_np)
#                 db_image_features = perform_inference(model, db_image_processed)
#                 if db_image_features is not None:
#                     match_found = compare_images(new_image_features.flatten(), db_image_features.flatten())
#                     if match_found:
#                         return db_image.user
#             return None
#         else:
#             return None
#     except Exception as e:
#         print("Error checking image match:", e)
#         return None

# new_image_path = 'C:/Users/mamle/OneDrive/Documents/Desktop/Backend/facedata/16.PNG'
# matched_user = check_image_match(new_image_path)
# if matched_user:
#     print(f"Match found! User: {matched_user}")
# else:
#     print("No match found.")

###########################################################################################3
# Load the pre-trained face recognition model
# current_directory = os.path.dirname(os.path.abspath(__file__))
# model_filename = 'newmodel.h5'
# model_path = os.path.join(current_directory, model_filename)

# # Load the model
# face_recognizer = load_model(model_path)
# @csrf_exempt
# def mark_attendance(request):
#     if request.method == 'POST' and request.FILES['image']:
#         image = request.FILES['image']
#         known_face_encodings = []
#         known_user_ids = []

#         # Load known face encodings and user IDs from database
#         face_data_objects = FaceData.objects.all()
#         for face_data_object in face_data_objects:
#             face_encoding_path = face_data_object.face_encoding.path
#             user_id = face_data_object.user_id
#             face_encoding = np.load(face_encoding_path)
#             known_face_encodings.append(face_encoding)
#             known_user_ids.append(user_id)

#         # Load the uploaded image
#         image_bytes = image.read()
#         nparr = np.fromstring(image_bytes, np.uint8)
#         image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Preprocess the image for the model
#         # You may need to resize, normalize, or preprocess the image according to your model requirements
#         processed_image = preprocess_image(image_np)

#         # Use the pre-trained model for face recognition
#         predicted_encoding = face_recognizer.predict(np.expand_dims(processed_image, axis=0))

#         # Compare the predicted encoding with known face encodings
#         # You may need to define a similarity threshold and perform a distance comparison here
#         # For simplicity, assuming a simple comparison here
#         if True:  # Add your condition for a match
#             matched_user_id = known_user_ids[0]  # Assuming the first user for simplicity
#             student = Student.objects.get(user_id=matched_user_id)
#             today_date = datetime.now().date()
#             attendance, created = Attendance.objects.get_or_create(date=today_date, student=student)
#             attendance.status = 'Present'
#             attendance.save()
#             return JsonResponse({'status': 'Attendance marked as Present'})
#         else:
#             today_date = datetime.now().date()
#             attendance, created = Attendance.objects.get_or_create(date=today_date, student=None)
#             attendance.status = 'Absent'
#             attendance.save()
#             return JsonResponse({'status': 'No match found, attendance marked as Absent'})
#     else:
#         return JsonResponse({'error': 'Please upload an image'})

# def preprocess_image(image):
#     try:
#         if image is not None:
#             print("Original image shape:", image.shape)  # Debugging statement
#             resized_image = cv2.resize(image, (224, 224))
#             print("Resized image shape:", resized_image.shape)  # Debugging statement
#             normalized_image = resized_image.astype('float32') / 255.0
#             processed_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
#             print("Processed image shape (after adding batch dimension):", processed_image.shape)  # Debugging statement
#             return processed_image
#         else:
#             raise ValueError("Input image is None")
#     except Exception as e:
#         print("Error preprocessing image:", e)
#         return None
# Define your class labels heredef extract_embeddings_from_database():
 
# Load CLASS_LABELS and MODEL_PATH###########################################
# CLASS_LABELS = ["johncend", "Hana chanie", "Hamere", "Dave", "Eve"]
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'newmodel.h5')

# @csrf_exempt
# @api_view(["POST"])
# def recognize_face(request):
#     if 'image' not in request.FILES:
#         return JsonResponse({'error': 'No image uploaded'}, status=400)

#     image_file = request.FILES['image']
#     file_path = os.path.join(settings.MEDIA_ROOT, image_file.name)

#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     # Save the uploaded image file
#     with open(file_path, 'wb+') as destination:
#         for chunk in image_file.chunks():
#             destination.write(chunk)

#     # Load face encodings from the database
#     known_encodings, known_labels = load_face_encodings_from_database()

#     # Load and encode the input image
#     input_image = face_recognition.load_image_file(file_path)
#     input_encoding = face_recognition.face_encodings(input_image)[0]

#     # Compare face encodings with the encodings from the database
#     matches = face_recognition.compare_faces(known_encodings, input_encoding)
#     matched_index = None
#     if True in matches:
#         matched_index = matches.index(True)
#         matched_user = known_labels[matched_index]
#         return JsonResponse({'matched_user': matched_user})
#     else:
#         return JsonResponse({'message': 'No match found'})

# def load_face_encodings_from_database():
#     encodings = []
#     labels = []
#     faces = FaceData.objects.all()
#     for face in faces:
#         # Load face encodings from the database
#         encoding = np.load(face.face_encoding.path)
#         encodings.append(encoding)
#         labels.append(face.user.username)
#     return encodings, labels
#####################################################################
def create_dataset(username):
    id = username
    if not os.path.exists(f'face_recognition_data/training_dataset/{id}/'):
        os.makedirs(f'face_recognition_data/training_dataset/{id}/')
    directory = f'face_recognition_data/training_dataset/{id}/'

    # Loading the face detector and the shape predictor for alignment
    print("[INFO] Loading the facial detector")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the video stream
    print("[INFO] Initializing Video stream")
    vs = cv2.VideoCapture(0)

    sampleNum = 0
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_aligned = gray_frame[y:y+h, x:x+w]
            sampleNum = sampleNum + 1

            cv2.imwrite(f'{directory}/{sampleNum}.jpg', face_aligned)
            face_aligned = imutils.resize(face_aligned, width=400)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.waitKey(50)

        cv2.imshow("Add Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if sampleNum > 300:
            break

    vs.release()
    cv2.destroyAllWindows()

# Load the pre-trained face recognition model
face_recognition_model = load_model('newmodel.h5')

def predict(face_aligned, threshold=0.7):
    processed_face = preprocess_image(face_aligned)
    face_embeddings = face_recognition_model.predict(np.expand_dims(processed_face, axis=0))

    # Compare face embeddings with the embeddings from the database
    embeddings_database = extract_embeddings_from_database()  # Assuming you have a function for this
    matched_index = compare_faces(embeddings_database, face_embeddings)

    if matched_index is not None:
        matched_user = FaceData.objects.all()[matched_index].user
        return (matched_user.username, 1.0)  # Returning username and confidence score
    else:
        return ('Unknown', 0.0)  # Returning 'Unknown' label and confidence score

def preprocess_image(face_aligned):
    # Resize the image to the required input shape of the model (e.g., 96x96)
    resized_image = cv2.resize(face_aligned, (96, 96))
    
    # Normalize the pixel values to be in the range [0, 1]
    processed_image = resized_image.astype('float32') / 255.0
    
    # If needed, you can apply additional preprocessing steps here
    
    return processed_image


# Other functions remain unchanged

def vizualize_Data(embedded, targets):
    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1))
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
    plt.close()