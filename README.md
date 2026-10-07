LocalTaste
A locality-aware restaurant recommendation and review sentiment analysis system built with machine learning.

Overview
LocalTaste combines two machine learning components:

Restaurant recommender – ranks restaurants in a selected locality according to similarity with a user's preferred cuisines.

Review sentiment classifier – predicts whether a restaurant review is positive or negative.

The project uses a content-based recommender with TF-IDF and K-Nearest Neighbours (KNN), and a Complement Naive Bayes classifier for sentiment analysis. A FastAPI backend exposes the models through REST endpoints, while a React + Vite frontend presents the results.

Features
Locality-based restaurant filtering.

Cuisine similarity ranking using TF-IDF and cosine distance.

Top-K restaurant recommendations.

Sentiment prediction for customer reviews.

Deduplicated restaurant and review data.

FastAPI endpoints for recommendations and sentiment analysis.

SQLite database containing cleaned project data.

React + Vite frontend.

Saved trained models and vectorizers using Joblib.

Dataset
The project uses the Zomato Bangalore Restaurants Dataset from Kaggle.

The raw dataset contains 51,717 rows and 17 columns, including restaurant names, locations, cuisines, ratings, votes, approximate cost, restaurant type, popular dishes, and customer reviews.

Processed data
Dataset	Before	After	Reduction
Restaurants	41,654	9,245	77.8%
Labelled reviews	1,025,753	108,858	89.4%
Duplicate listings were removed using the (name, locality) key. This prevents nearly identical restaurant entries and reviews from appearing across both training and testing data.

Machine Learning Components
Restaurant recommender
The recommender follows these steps:

Restaurant cuisine fields are cleaned and converted into TF-IDF vectors.

Restaurants are grouped by locality.

A separate KNN model is maintained for each locality.

The user's cuisine preference or previously liked restaurants are converted into the same TF-IDF representation.

Cosine distance is used to measure cuisine similarity.

Restaurants are ranked and the top-K results are returned.

This content-based approach avoids the cold-start problem associated with collaborative filtering when user-rating history is limited.

Sentiment classifier
Review sentiment labels are derived from star ratings:

Rating >= 4: positive.

Rating <= 2: negative.

Ratings from 2.5 to 3.5: excluded as ambiguous.

The final labelled dataset contains 108,858 reviews:

81,162 positive reviews.

27,696 negative reviews.

The data is split using an 80/20 stratified split with random_state=42. Reviews are transformed using TF-IDF with up to 10,000 features, English stop-word removal, and a minimum document frequency of 5.

Two Naive Bayes models were compared:

MultinomialNB baseline.

ComplementNB selected model.

ComplementNB was selected because it achieved much higher negative-class recall, which is useful for identifying genuine complaints despite a small reduction in overall accuracy.

Metric	MultinomialNB	ComplementNB
Accuracy	92.38%	92.33%
Weighted precision	92.46%	92.72%
Weighted recall	92.38%	92.33%
Weighted F1-score	92.08%	92.45%
Negative-class precision	94%	82%
Negative-class recall	75%	90%
Negative-class F1-score	0.83	0.86
System Architecture
text
User
  |
  v
React + Vite Frontend
  |
  v
FastAPI Backend
  |----------------------|
  v                      v
Recommendation API       Sentiment API
  |                      |
  v                      v
TF-IDF + KNN              TF-IDF + ComplementNB
  |                      |
  v                      v
SQLite restaurant data   Saved Joblib model/vectorizer
API Endpoints
POST /api/recommend
Returns restaurants that are similar to the user's selected restaurants or cuisine preference within a locality.

Example request:

json
{
  "locality": "Banashankari",
  "liked_names": ["Jalsa"],
  "top_k": 5
}
Example response:

json
[
  {
    "name": "BOX8 - Desi Meals",
    "cuisines": "North Indian, Mughlai, Fast Food",
    "similarity": 0.833
  },
  {
    "name": "Vaishali Deluxe",
    "cuisines": "North Indian, Chinese",
    "similarity": 0.475
  }
]
POST /api/sentiment
Predicts the sentiment of a customer review using the trained Complement Naive Bayes model.

Example request:

json
{
  "review": "The food was excellent and the service was very quick."
}
Example response:

json
{
  "sentiment": "positive"
}
Project Structure
The exact structure may vary depending on the implementation, but the project is organized around the following components:

text
LocalTaste/
├── backend/              # FastAPI application and API routes
├── frontend/             # React + Vite user interface
├── data/                 # Cleaned restaurant and review data
├── models/               # Saved ML models and vectorizers
├── notebooks/            # Data exploration and experiments
├── scripts/              # Preprocessing and training scripts
├── requirements.txt      # Python dependencies
└── README.md
Installation
Prerequisites
Python 3.9 or later.

Node.js and npm.

A local SQLite installation or Python's built-in SQLite support.

The Zomato Bangalore Restaurants Dataset.

Clone the repository
bash
git clone https://github.com/Anup-raj2002/LocalTaste.git
cd LocalTaste
Set up the Python environment
bash
python -m venv .venv
Activate the environment on Linux or macOS:

bash
source .venv/bin/activate
Activate the environment on Windows:

powershell
.venv\Scripts\activate
Install the Python dependencies:

bash
pip install -r requirements.txt
Install frontend dependencies
bash
cd frontend
npm install
cd ..
Running the Project
Start the backend
From the project root, run:

bash
uvicorn backend.main:app --reload
The FastAPI service should be available at:

text
http://127.0.0.1:8000
Interactive API documentation is available at:

text
http://127.0.0.1:8000/docs
Start the frontend
bash
cd frontend
npm run dev
Open the local URL displayed by Vite in your browser.

Training the Models
A typical training workflow is:

Load the raw Kaggle dataset.

Clean ratings, cost, cuisine, and locality fields.

Parse the stringified reviews_list field using ast.literal_eval().

Explode the reviews into individual records.

Repair corrupted text using ftfy.

Deduplicate restaurants and reviews.

Derive positive and negative sentiment labels.

Train the recommender and sentiment classifier.

Evaluate both Naive Bayes variants.

Save the selected model and vectorizer with Joblib.

Example command pattern:

bash
python scripts/preprocess.py
python scripts/train.py
Update these commands if the project uses different script names.

Reproducibility
The sentiment classifier uses:

Stratified 80/20 train-test split.

random_state=42.

The same TF-IDF transformation for training and inference.

Saved model artifacts for later predictions.

Limitations
Recommendations are based on cuisine text and locality, not user behaviour or historical ratings.

The sentiment labels are inferred from star ratings and may not always reflect the actual wording of a review.

Neutral reviews are excluded from training because ratings from 2.5 to 3.5 are treated as ambiguous.

The dataset is specific to Bangalore restaurants and may contain outdated listings.

Similarity scores indicate cuisine similarity, not guaranteed food quality.

Future Improvements
Add user profiles and collaborative filtering when sufficient interaction data becomes available.

Include restaurant ratings, cost, votes, and opening status in the ranking model.

Add a neutral sentiment class.

Compare ComplementNB with Logistic Regression, Linear SVM, and transformer-based models.

Add model monitoring and periodic retraining.

Improve API validation, authentication, and deployment configuration.

Deploy the backend and frontend using Docker or a cloud platform.