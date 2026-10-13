# Smart Agriculture Crop Recommender (Data Mining / ML)

Recommends the most suitable crop from rainfall, average temperature and soil type.

Algorithms: Decision Tree (crop classification), Polynomial Ridge Regression, one model per crop (yield prediction), K-Means (agro-climate zones).

## Run
    python -m venv venv
    venv\Scripts\activate        (Windows)   |   source venv/bin/activate   (Mac/Linux)
    pip install -r requirements.txt
    python app.py
Open http://127.0.0.1:5000

The models train automatically on first run (or run `python train_model.py`).
Pre-generated `data/crop_data.csv` and `models/` are already included.

## Structure
    app.py            Flask web app + recommendation logic
    train_model.py    dataset generation + Decision Tree, Regression, K-Means training
    templates/        index.html
    static/           style.css
    data/             crop_data.csv
    models/           trained models + metrics.json
    API: /api/recommend?rainfall=1200&temp=26&soil=Clay
