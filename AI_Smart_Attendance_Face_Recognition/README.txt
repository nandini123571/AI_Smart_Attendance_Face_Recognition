
AI-BASED SMART ATTENDANCE USING FACE RECOGNITION
================================================

1. Install Python 3.10 or newer.
2. Open Command Prompt inside this project folder.
3. Run:
   pip install -r requirements.txt

4. Register a student:
   python app.py
   Choose 1
   Enter Roll/Student ID and Name.
   Look at the camera until 30 samples are captured.

5. Train:
   python app.py
   Choose 2

6. Take attendance:
   python app.py
   Choose 3
   The camera recognizes registered students and stores attendance.

7. View report:
   python app.py
   Choose 4

Database file: attendance.db
Face images: dataset/
Trained model: trainer.yml

IMPORTANT:
- Use this only with consent from students.
- Recognition accuracy depends on lighting, camera quality and training images.
- If the camera does not open, close other apps using the webcam.
