import glob
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

files = glob.glob('faces/*.jpeg')
files = [f for f in files if '_labeled' not in f]

known_faces = {}
face_id = 0

# Define font size
font = ImageFont.truetype("Arial.ttf", 30)

for file in files:
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            face_id = list(known_faces.keys())[first_match_index]
            print(f"face id {face_id} detected")
        else:
            face_id += 1
            known_faces[face_id] = face_encoding
            print(f"new face id {face_id}")

        # Label
        thickness = 3
        label_height = 45
        for i in range(thickness):
            draw.rectangle(((left - i, top - i), (right + i, bottom + i)), outline=(0, 0, 255))
        draw.rectangle(((left, bottom), (right, bottom + label_height)), fill=(0, 0, 255))
        draw.text((left + 6, bottom + 5), str(face_id), fill=(255, 255, 255, 255), font=font)

    del draw

    pil_image.save(f'faces/{file.split("/")[-1]}'.replace(".jpeg", "_labeled.jpeg"))