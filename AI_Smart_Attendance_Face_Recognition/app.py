
import cv2, os, sqlite3
from datetime import datetime

DB = "attendance.db"
DATASET = "dataset"
TRAINER = "trainer.yml"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATASET, exist_ok=True)

def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY, name TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER, name TEXT, date TEXT, time TEXT)""")
    con.commit()
    return con

def register():
    name = input("Enter student name: ").strip()
    sid = input("Enter roll/student ID: ").strip()
    if not name or not sid.isdigit():
        print("Invalid name or ID.")
        return
    sid = int(sid)
    con = db()
    con.execute("INSERT OR REPLACE INTO students(id,name) VALUES(?,?)", (sid,name))
    con.commit(); con.close()

    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(CASCADE)
    count = 0
    print("Look at the camera. Press q to stop.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            count += 1
            cv2.imwrite(f"{DATASET}/User.{sid}.{count}.jpg", gray[y:y+h,x:x+w])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"Samples: {count}/30",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Registration", frame)
        if cv2.waitKey(1) & 0xff == ord('q') or count >= 30: break
    cap.release(); cv2.destroyAllWindows()
    print("Registration completed.")

def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(CASCADE)
    faces, ids = [], []
    for fn in os.listdir(DATASET):
        if not fn.lower().endswith(".jpg"): continue
        try: sid = int(fn.split(".")[1])
        except: continue
        img = cv2.imread(os.path.join(DATASET,fn), cv2.IMREAD_GRAYSCALE)
        faces.append(img); ids.append(sid)
    if not faces:
        print("No face samples found. Register a student first."); return
    recognizer.train(faces, __import__("numpy").array(ids))
    recognizer.write(TRAINER)
    print("Model trained successfully.")

def mark_attendance(sid, name):
    con = db()
    today = datetime.now().strftime("%Y-%m-%d")
    cur = con.execute("SELECT 1 FROM attendance WHERE student_id=? AND date=?",(sid,today))
    if cur.fetchone() is None:
        now = datetime.now()
        con.execute("INSERT INTO attendance(student_id,name,date,time) VALUES(?,?,?,?)",
                    (sid,name,today,now.strftime("%H:%M:%S")))
        con.commit()
        print(f"Attendance marked: {name}")
    con.close()

def attendance():
    if not os.path.exists(TRAINER):
        print("Train the model first."); return
    con = db()
    students = {r[0]:r[1] for r in con.execute("SELECT id,name FROM students")}
    con.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER)
    detector = cv2.CascadeClassifier(CASCADE)
    cap = cv2.VideoCapture(0)
    print("Attendance camera started. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.2,5)
        for (x,y,w,h) in faces:
            sid, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # LBPH confidence: lower is better. 70 is a reasonable starter threshold.
            if confidence < 70 and sid in students:
                name = students[sid]
                mark_attendance(sid,name)
                label = f"{name} ({sid})"
            else:
                label = "Unknown"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
        cv2.imshow("Smart Attendance",frame)
        if cv2.waitKey(1)&0xff==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def report():
    con = db()
    rows = con.execute("SELECT student_id,name,date,time FROM attendance ORDER BY date DESC,time DESC").fetchall()
    con.close()
    print("\n--- ATTENDANCE REPORT ---")
    for r in rows: print(r)
    print("-------------------------")

if __name__ == "__main__":
    db()
    print("\nAI-BASED SMART ATTENDANCE")
    print("1. Register Student")
    print("2. Train Model")
    print("3. Start Attendance")
    print("4. View Report")
    choice = input("Choose: ").strip()
    if choice=="1": register()
    elif choice=="2": train()
    elif choice=="3": attendance()
    elif choice=="4": report()
    else: print("Invalid choice.")
