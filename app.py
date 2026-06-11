from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import sqlite3

from datetime import datetime

from collections import Counter

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# =========================
# DATABASE
# =========================
DATABASE = 'fungi.db'
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# CREATE TABLE AUTOMATICALLY
conn = get_db_connection()
conn.execute('''
             
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image TEXT,
    fungus TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
)
''')
conn.commit()
conn.close()


# =========================
# CONFIGURATION
# =========================
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# LOAD AI MODEL
# =========================
model = load_model('model/fungi_mobilenet_pertype.h5')
classes = [
    'Candida albicans',
    'Epidermophyton floccosum',
    'Trichophyton mentagrophytes',
    'Trichophyton rubrum'
]


# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('home.html')


# =========================
# ABOUT PAGE
# =========================
@app.route('/about')
def about():
    return render_template('about.html')


# =========================
# HISTORY PAGE
# =========================
@app.route('/history')
def history():
    search = request.args.get('search', '')
    filter_value = request.args.get('filter', 'All')
    conn = get_db_connection()
    query = "SELECT * FROM history WHERE 1=1"
    params = []

    # SEARCH
    if search:
        query += " AND fungus LIKE ?"
        params.append(f"%{search}%")

    # FILTER
    if filter_value != 'All':
        query += " AND fungus = ?"
        params.append(filter_value)
    query += " ORDER BY id DESC"
    data = conn.execute(query, params).fetchall()
    conn.close()
    return render_template(
        'history.html',
        data=data
    )


@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("fungi.db")

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # =========================
    # GET ALL HISTORY
    # =========================

    cursor.execute("""

        SELECT *

        FROM history

        ORDER BY created_at DESC

    """)

    rows = cursor.fetchall()

    # =========================
    # TOTAL
    # =========================

    total_predictions = len(rows)
    
    fungus_list = []

    for row in rows:
        fungus_list.append(row["fungus"])
    if fungus_list:
        top_fungus = Counter(fungus_list).most_common(1)[0][0]
    else:
        top_fungus = "N/A"

    # NOW 5 CLASSES
    fungus_classes = 4

    # =========================
    # COUNTS
    # =========================

    candida_count = 0

    epidermo_count = 0

    mentagrophytes_count = 0

    rubrum_count = 0

    unknown_count = 0

    # =========================
    # CONFIDENCE STORAGE
    # =========================

    candida_conf = []

    epidermo_conf = []

    mentagrophytes_conf = []

    rubrum_conf = []

    unknown_conf = []

    all_confidence = []

    # =========================
    # LOOP THROUGH DATA
    # =========================

    for row in rows:

        fungus = row["fungus"]

        confidence = float(row["confidence"])

        all_confidence.append(confidence)

        # =========================
        # CANDIDA
        # =========================

        if fungus == "Candida albicans":

            candida_count += 1

            candida_conf.append(confidence)

        # =========================
        # EPIDERMOPHYTON
        # =========================

        elif fungus == "Epidermophyton floccosum":

            epidermo_count += 1

            epidermo_conf.append(confidence)

        # =========================
        # MENTAGROPHYTES
        # =========================

        elif fungus == "Trichophyton mentagrophytes":

            mentagrophytes_count += 1

            mentagrophytes_conf.append(confidence)

        # =========================
        # RUBRUM
        # =========================

        elif fungus == "Trichophyton rubrum":

            rubrum_count += 1

            rubrum_conf.append(confidence)

        # =========================
        # UNKNOWN
        # =========================

        elif fungus == "Unknown":

            unknown_count += 1

            unknown_conf.append(confidence)

    # =========================
    # AVERAGE FUNCTION
    # =========================

    def avg(data):

        if len(data) == 0:

            return 0

        return round(sum(data) / len(data), 2)

    # =========================
    # AVERAGE VALUES
    # =========================

    candida_avg = avg(candida_conf)

    epidermo_avg = avg(epidermo_conf)

    mentagrophytes_avg = avg(
        mentagrophytes_conf
    )

    rubrum_avg = avg(rubrum_conf)

    unknown_avg = avg(unknown_conf)

    # =========================
    # OVERALL ACCURACY
    # =========================

    overall_accuracy = avg(all_confidence)

    # =========================
    # HIGH CONFIDENCE CASES
    # =========================

    high_confidence_cases = len(

        [c for c in all_confidence if c >= 90]

    )

    # =========================
    # TIMELINE DATA
    # =========================

    timeline_data = {}

    for row in rows:

        date_only = str(
            row["created_at"]
        ).split(" ")[0]

        if date_only not in timeline_data:

            timeline_data[date_only] = 0

        timeline_data[date_only] += 1

    timeline_labels = list(
        timeline_data.keys()
    )

    timeline_values = list(
        timeline_data.values()
    )

    # =========================
    # RECENT ROWS
    # =========================

    recent_rows = rows[:5]

    conn.close()

    # =========================
    # RENDER TEMPLATE
    # =========================
    current_date = datetime.now().strftime("%d %B %Y")
    return render_template(

    "dashboard.html",

    top_fungus=top_fungus,

    current_date=current_date,

    # TOTAL
    total_predictions=total_predictions,

    # CLASSES
    fungus_classes=fungus_classes,

    # ACCURACY
    overall_accuracy=overall_accuracy,

    # HIGH CONFIDENCE
    high_confidence_cases=high_confidence_cases,

    # COUNTS
    candida_count=candida_count,

    epidermo_count=epidermo_count,

    mentagrophytes_count=mentagrophytes_count,

    rubrum_count=rubrum_count,

    unknown_count=unknown_count,

    # AVERAGE CONFIDENCE
    candida_avg=candida_avg,

    epidermo_avg=epidermo_avg,

    mentagrophytes_avg=mentagrophytes_avg,

    rubrum_avg=rubrum_avg,

    unknown_avg=unknown_avg,

    # TIMELINE
    timeline_labels=timeline_labels,

    timeline_values=timeline_values,

    # RECENT
    recent_rows=recent_rows

)
    
# =========================
# PREDICTION PAGE
# =========================
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


# =========================
# RESULT PAGE
# =========================
@app.route('/result')
def result():
    return render_template('result.html')


# =========================
# UPLOAD & PREDICTION
# =========================
@app.route('/upload', methods=['POST'])
def upload():
    # CHECK IMAGE
    if 'image' not in request.files:
        return "No File Uploaded"
    file = request.files['image']

    # EMPTY FILE
    if file.filename == '':
        return "No Selected File"

    # SAVE FILE
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )
    file.save(filepath)

    # =========================
    # IMAGE PROCESSING
    # =========================
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0


    # =========================
    # AI PREDICTION
    # =========================
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = round(
        float(np.max(prediction)) * 100,
        2
    )


    # =========================
    # UNKNOWN DETECTION
    # =========================
    if confidence < 70:
        predicted_class = "Unknown"
        confidence = 0
    else:
        predicted_class = classes[predicted_index]


    # =========================
    # FACT DATABASE
    # =========================
    fungi_facts = {
        "Candida albicans": {
            "quick": "Candida albicans is a common fungal pathogen in humans.",
            "detail": """
Candida albicans is a yeast-like fungus normally present in the human body.
It may cause infections such as oral thrush, vaginal candidiasis, and bloodstream infections
when the immune system becomes weakened.
"""
        },

        "Epidermophyton floccosum": {
            "quick": "Dermatophyte fungus causing skin and nail infections.",
            "detail": """
Epidermophyton floccosum is a dermatophyte responsible for athlete’s foot,
ringworm, and nail infections. It spreads through direct contact and thrives
in warm and humid environments.
"""
        },

        "Trichophyton mentagrophytes": {
            "quick": "Common dermatophyte affecting skin, hair, and nails.",
            "detail": """
Trichophyton mentagrophytes causes fungal infections such as athlete's foot,
ringworm, and scalp infections. It spreads easily in moist environments and
through infected surfaces or animals.
"""
        },

        "Trichophyton rubrum": {
            "quick": "One of the most common dermatophyte fungi worldwide.",
            "detail": """
Trichophyton rubrum is a dermatophyte fungus responsible for skin, nail,
and foot infections. It is one of the most common causes of athlete’s foot
and chronic fungal nail disease worldwide.
"""
        }
    }


    # =========================
    # UNKNOWN FACT
    # =========================
    if predicted_class == "Unknown":
        quick_fact = "No fungi detected from trained AI dataset."
        detail_fact = """
The uploaded image could not be classified because it does not match
the fungi classes used during AI model training.
Please upload a clearer fungi microscopic image.
"""
    else:
        quick_fact = fungi_facts[predicted_class]["quick"]
        detail_fact = fungi_facts[predicted_class]["detail"]
        
        
    # =========================
    # SAVE DATABASE
    # =========================
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO history
        (image, fungus, confidence)
        VALUES (?, ?, ?)
        """,
        (
            file.filename,
            predicted_class,
            confidence
        )
    )
    conn.commit()
    conn.close()


    # =========================
    # RETURN RESULT
    # =========================
    return render_template(
        'result.html',
        image_path=filepath,
        prediction=predicted_class,
        confidence=confidence,
        quick_fact=quick_fact,
        detail_fact=detail_fact
    )


# =========================
# DELETE RECORD
# =========================
@app.route('/delete/<int:record_id>')
def delete_record(record_id):
    conn = get_db_connection()
    conn.execute(
        "DELETE FROM history WHERE id=?",
        (record_id,)
    )
    conn.commit()
    conn.close()
    return redirect ('/history')


# =========================
# DOWNLOAD REPORT
# =========================
@app.route('/download/<int:record_id>')
def download_file(record_id):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM history WHERE id=?",
        (record_id,)
    ).fetchone()
    conn.close()
    if not row:
        return "Record not found"
    prediction = row["fungus"]
    confidence = row["confidence"]
    image_path = "static/uploads/" + row["image"]

    # =========================
    # FACT DATABASE
    # =========================
    fungi_facts = {
        "Candida albicans": {
            "quick": "Candida albicans is a common fungal pathogen in humans.",
            "detail": """
Candida albicans is a yeast-like fungus normally present in the human body.
It may cause infections such as oral thrush, vaginal candidiasis, and bloodstream infections
when the immune system becomes weakened.
"""
        },

        "Epidermophyton floccosum": {
            "quick": "Dermatophyte fungus causing skin and nail infections.",
            "detail": """
Epidermophyton floccosum is a dermatophyte responsible for athlete’s foot,
ringworm, and nail infections. It spreads through direct contact and thrives
in warm and humid environments.
"""
        },
        "Trichophyton mentagrophytes": {
            "quick": "Common dermatophyte affecting skin, hair, and nails.",
            "detail": """
Trichophyton mentagrophytes causes fungal infections such as athlete's foot,
ringworm, and scalp infections. It spreads easily in moist environments and
through infected surfaces or animals.
"""
        },
        "Trichophyton rubrum": {
            "quick": "One of the most common dermatophyte fungi worldwide.",
            "detail": """
Trichophyton rubrum is a dermatophyte fungus responsible for skin, nail,
and foot infections. It is one of the most common causes of athlete’s foot
and chronic fungal nail disease worldwide.
"""
        }
    }

    # =========================
    # UNKNOWN
    # =========================
    if prediction == "Unknown":
        quick_fact = "No fungi detected from trained AI dataset."
        detail_fact = """
The uploaded image could not be classified because it does not match
the fungi classes used during AI model training.
Please upload a clearer fungi microscopic image.
"""
    else:
        quick_fact = fungi_facts[prediction]["quick"]
        detail_fact = fungi_facts[prediction]["detail"]

    # =========================
    # RETURN RESULT PAGE
    # =========================
    return render_template(
        'result.html',
        image_path=image_path,
        prediction=prediction,
        confidence=confidence,
        quick_fact=quick_fact,
        detail_fact=detail_fact
    )

# =========================
# DELETE MULTIPLE RECORDS
# =========================

@app.route("/delete-multiple")
def delete_multiple():

    ids = request.args.get("ids")

    if ids:

        id_list = ids.split(",")

        conn = get_db_connection()

        for record_id in id_list:

            conn.execute(
                "DELETE FROM history WHERE id=?",
                (record_id,)
            )

        conn.commit()

        conn.close()

    return redirect(url_for("history"))

# =========================
# ENCYCLOPEDIA
# =========================
@app.route('/encyclopedia')
def encyclopedia():
    return render_template('encyclopedia.html')

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)
