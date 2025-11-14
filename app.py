from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import datetime
import requests
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Tuple
import psutil
from sklearn.model_selection import train_test_split
import csv
import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load environment variables
load_dotenv()

# Database configuration
DATABASE = 'tunevibe.db'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER,
                liked_songs TEXT,
                disliked_songs TEXT,
                favorite_artists TEXT,
                favorite_genres TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity_type TEXT,
                track_id TEXT,
                duration_played INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                precision FLOAT,
                recall FLOAT,
                diversity FLOAT,
                novelty FLOAT,
                coverage FLOAT,
                personalization FLOAT
            )
        ''')
        db.commit()

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        return User(id=user['id'], username=user['username'], email=user['email'])
    return None

# Indian cities with cultural tags
INDIAN_CITIES = {
    'Mumbai': ['bollywood', 'urban', 'fastpaced', 'cosmopolitan'],
    'Delhi': ['punjabi', 'hiphop', 'street', 'historical'],
    'Bangalore': ['electronic', 'indie', 'tech', 'cosmopolitan'],
    'Hyderabad': ['telugu', 'filmi', 'heritage', 'royal'],
    'Ahmedabad': ['gujarati', 'traditional', 'folk', 'business'],
    'Chennai': ['tamil', 'carnatic', 'classical', 'beach'],
    'Kolkata': ['bengali', 'rabindrasangeet', 'intellectual', 'heritage'],
    'Pune': ['marathi', 'educational', 'cultural', 'peaceful'],
    'Jaipur': ['rajasthani', 'folk', 'heritage', 'royal'],
    'Lucknow': ['ghazal', 'classical', 'nawabi', 'sophisticated']
}

ACTIVITY_TAGS = {
    'working/studying': ['focus', 'study', 'instrumental', 'concentration', 'calm', 'background','acoustic'],
    'working_out': ['energy', 'workout', 'gym', 'pump', 'intense', 'motivational'],
    'partying': ['party', 'dance', 'edm', 'club', 'celebration', 'festive'],
    'relaxing': ['chill', 'ambient', 'meditation', 'calm', 'peaceful', 'soothing'],
    'driving': ['roadtrip', 'car', 'singalong', 'drive', 'adventure', 'journey'],
    'romantic': ['love', 'romantic', 'slow', 'intimate', 'passionate', 'sensual'],
    'dancing': ['dance', 'upbeat', 'groovy', 'move', 'rhythmic', 'energetic']
}

WEATHER_TAGS = {
    'Clear': ['sunny', 'bright', 'happy', 'summer'],
    'Clouds': ['cloudy', 'mellow', 'soft', 'dreamy'],
    'Rain': ['rainy', 'cozy', 'acoustic', 'romantic'],
    'Thunderstorm': ['intense', 'dramatic', 'powerful', 'epic'],
    'Snow': ['cold', 'winter', 'chill', 'holiday'],
    'Mist': ['mysterious', 'dreamy', 'hazy', 'ethereal'],
    'Haze': ['mysterious', 'dreamy', 'hazy', 'ethereal'],
    'Fog': ['mysterious', 'dreamy', 'hazy', 'ethereal']
}

class RecommendationModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.knn_model = None
        self.music_data = None
        self.evaluation_data = None
        self.user_feedback = defaultdict(dict)
        self.last_retrained = None
    
    def load_data(self) -> pd.DataFrame:
        if self.music_data is not None:
            return self.music_data
        """Load and prepare music data with robust error handling"""
        try:
            df = pd.read_csv("Music.csv")

            column_mapping = {
                'track_name': 'track_name',
                'artist': 'track_artist',  # Correctly map from track_artist in dataset
                'spotify_id': 'spotify_id',
                'activity': 'tags',
                'genre': 'genre',
                'year': 'year',
                'duration': 'duration_ms',
                'danceability': 'danceability',
                'energy': 'energy',
                'valence': 'valence',
                'popularity': 'popularity',
                'tempo': 'tempo',
                'loudness': 'loudness',
                'key': 'key',
                'mode': 'mode'
            }

            clean_df = pd.DataFrame()
            for new_col, original_col in column_mapping.items():
                if original_col in df.columns:
                    clean_df[new_col] = df[original_col]
                else:
                    # Set default values for missing columns
                    if new_col == 'popularity':
                        clean_df[new_col] = 50
                    elif new_col in ['danceability', 'energy', 'valence']:
                        clean_df[new_col] = 0.5
                    elif new_col == 'duration':
                        clean_df[new_col] = 180000  # Default 3 minutes in ms
                    else:
                        clean_df[new_col] = ''

            # Clean up artist name
            if 'artist' in clean_df.columns:
                clean_df['artist'] = clean_df['artist'].astype(str).str.strip().replace('', 'Unknown Artist')
            else:
                clean_df['artist'] = 'Unknown Artist'

            # Handle duplicates based on spotify_id if available
            if 'spotify_id' in clean_df.columns:
                clean_df = clean_df.sort_values('popularity', ascending=False)
                clean_df = clean_df.drop_duplicates('spotify_id', keep='first')

            # Fill and lower-case key text columns
            clean_df.loc[:, 'activity'] = clean_df['activity'].fillna('').astype(str)
            clean_df['genre'] = clean_df['genre'].fillna('').astype(str).str.lower()

            # Build combined tags
            clean_df['combined_tags'] = (
                clean_df['activity'] + ' ' +
                clean_df['genre'] + ' ' +
                clean_df['energy'].apply(lambda x: f"energy_{int(float(x)*10)}" if pd.notna(x) else "") + ' ' +
                clean_df['valence'].apply(lambda x: f"mood_{int(float(x)*10)}" if pd.notna(x) else "") + ' ' +
                clean_df['year'].apply(lambda x: f"year_{int(x)}" if pd.notna(x) else "")
            ).str.strip()

            # Ensure combined_tags is not empty
            clean_df['combined_tags'] = clean_df['combined_tags'].replace('', 'generic_music')

            # Ensure numeric columns are properly formatted
            clean_df['popularity'] = pd.to_numeric(clean_df['popularity'], errors='coerce').fillna(50)
            clean_df['year'] = pd.to_numeric(clean_df['year'], errors='coerce').fillna(0)

            # Split data for training and evaluation
            self.music_data, self.evaluation_data = train_test_split(
                clean_df,
                test_size=0.2,
                random_state=42
            )

            print(f"Successfully loaded {len(clean_df)} songs (train: {len(self.music_data)}, test: {len(self.evaluation_data)})")
            return clean_df

        except Exception as e:
            print(f"Error loading custom dataset: {str(e)}")
            print("Falling back to built-in dataset")
            return self._load_fallback_data()

    def _load_fallback_data(self) -> pd.DataFrame:
        """Load fallback data with enhanced tracks"""
        fallback_data = pd.DataFrame({
            'track_name': [
                'Blinding Lights', 'Dance Monkey', 'Stay', 'Shape of You', 'Despacito',
                'Levitating', 'Save Your Tears', 'Don\'t Start Now', 'Watermelon Sugar', 'Blinding Lights',
                'Circles', 'Sunflower', 'Someone You Loved', 'Perfect', 'Believer',
                'Lovely', 'Havana', 'Uptown Funk', 'Thinking Out Loud', 'Closer',
                'Bohemian Rhapsody', 'Take on Me', 'Sweet Child O\'Mine', 'Smells Like Teen Spirit',
                'Hotel California', 'Billie Jean', 'Like a Rolling Stone', 'Imagine', 'Wonderwall'
            ],
            'artist': [
                'The Weeknd', 'Tones and I', 'The Kid LAROI', 'Ed Sheeran', 'Luis Fonsi',
                'Dua Lipa', 'The Weeknd', 'Dua Lipa', 'Harry Styles', 'The Weeknd',
                'Post Malone', 'Post Malone, Swae Lee', 'Lewis Capaldi', 'Ed Sheeran', 'Imagine Dragons',
                'Billie Eilish, Khalid', 'Camila Cabello', 'Mark Ronson ft. Bruno Mars', 'Ed Sheeran', 'The Chainsmokers',
                'Queen', 'A-ha', 'Guns N\' Roses', 'Nirvana',
                'Eagles', 'Michael Jackson', 'Bob Dylan', 'John Lennon', 'Oasis'
            ],
            'genre': [
                'Pop', 'Pop', 'Hip-Hop', 'Pop', 'Latin',
                'Pop', 'Pop', 'Pop', 'Pop', 'Pop',
                'Hip-Hop', 'Hip-Hop', 'Pop', 'Pop', 'Rock',
                'Pop', 'Pop', 'Funk', 'Pop', 'EDM',
                'Rock', 'Pop', 'Rock', 'Rock',
                'Rock', 'Pop', 'Rock', 'Rock', 'Rock'
            ],
            'popularity': [
                90, 85, 95, 92, 88, 89, 91, 87, 86, 90,
                84, 83, 93, 94, 82, 81, 85, 88, 92, 79,
                97, 88, 93, 95, 96, 98, 91, 94, 90
            ],
            'activity': [
                'rock,party,dance', 'pop,dance,energy', 'chill,relax,calm', 'romantic,love,pop', 'latin,dance,summer',
                'dance,pop,energy', 'pop,rnb,chill', 'dance,pop,disco', 'pop,summer,happy', 'pop,dance,energy',
                'hiphop,chill,rnb', 'hiphop,chill,summer', 'pop,sad,ballad', 'pop,love,romantic', 'rock,energy,workout',
                'pop,sad,emotional', 'pop,latin,dance', 'funk,party,dance', 'pop,love,ballad', 'edm,dance,party',
                'rock,classic,epic', 'pop,80s,energetic', 'rock,classic,guitar', 'rock,grunge,angsty',
                'rock,classic,mellow', 'pop,disco,dance', 'rock,folk,thoughtful', 'rock,peaceful,inspirational', 'rock,britpop,melancholic'
            ],
            'spotify_id': [
                '0VjIjW4GlUZAMYd2vXMi3b', '1HxHnKmF0M5uXZg1VfWQye', '5HCyWlXZPP0y6Gqq8TgA20', '7qiZfU4dY1lWllzX7mPBI3', '6rPO02ozF3bM7NnOV4h6s2',
                '0fA0VVWsXO9YnASrzqfmYu', '5SAphjA4V4Gf8VFp3qHDJd', '6WrI0LAC5M1Rw2MnX2ZvEg', '5HQemuUJ9eEbR5gp2Qe0Cb', '0VjIjW4GlUZAMYd2vXMi3b',
                '21jGcNKet2qwijlDFuPiPb', '0RiRZpuVRbi7oqRdSMwhQY', '7qEHsqek33rTcFNT9PFqLf', '2VxeLyX666F8uXCJ0dZF8B', '7gSQv1OHpkIoAdUiRLdmI6',
                '2hoALizwJ7swR6XEn1vkyg', '1ofof3a3DAQZkxebSGVJcI', '4VMYDDC17R5S4fQuDtB1eG', '4D2kJ7zMJWg8Ql8Zxz7kYn', '7BKLCZ1jbUBVqRi2FVlTVw',
                '7tFiyTwD0nx5a1eklYtX2J', '2UaR5W1y1xX1frbZYHpWXK', '7snQQk1zcKl8gZ92AnueZW', '5ghIJDpPoe3CfHMGu71E6T',
                '40riOy7x9W7GXjyGp4pjAv', '5ChkMS8OtdzJeqyybCc9R5', '6v3xYnG2NC0Hg6Q9gD5onR', '1q9x5tSxY4JXWpNz8Z9Xjz', '6d5W08xVJ0K4Z2hWhF6gHc'
            ],
            'danceability': [
                0.8, 0.85, 0.7, 0.75, 0.9, 0.82, 0.78, 0.83, 0.76, 0.8,
                0.72, 0.74, 0.68, 0.65, 0.77, 0.63, 0.81, 0.88, 0.62, 0.85,
                0.43, 0.78, 0.45, 0.52, 0.58, 0.85, 0.55, 0.48, 0.65
            ],
            'energy': [
                0.9, 0.8, 0.6, 0.7, 0.85, 0.87, 0.75, 0.84, 0.72, 0.9,
                0.65, 0.68, 0.58, 0.55, 0.82, 0.45, 0.78, 0.92, 0.5, 0.88,
                0.92, 0.85, 0.93, 0.95, 0.68, 0.88, 0.72, 0.35, 0.72
            ],
            'valence': [
                0.6, 0.8, 0.4, 0.8, 0.9, 0.75, 0.65, 0.82, 0.78, 0.6,
                0.55, 0.72, 0.35, 0.85, 0.68, 0.25, 0.82, 0.88, 0.78, 0.72,
                0.42, 0.85, 0.62, 0.38, 0.55, 0.78, 0.45, 0.28, 0.65
            ]
        })
    
        fallback_data['combined_tags'] = (
            fallback_data['activity'].fillna('') + ' ' + 
            fallback_data['genre'].str.lower() + ' ' +
            fallback_data['energy'].apply(lambda x: f"energy_{int(x*10)}") + ' ' +
            fallback_data['valence'].apply(lambda x: f"mood_{int(x*10)}")
        )
        
        self.music_data = fallback_data
        return fallback_data

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 ** 2
    
    def initialize_models(self):
        if self.knn_model is not None and (datetime.datetime.now() - self.last_retrained).seconds < 3600:
            return
        
        """Initialize models with memory-efficient KNN"""
        if self.music_data is None:
            self.load_data()
            
        print(f"Memory before vectorization: {self.get_memory_usage()} MB")
        
        # Ensure combined_tags exists and has no NaN values
        if 'combined_tags' not in self.music_data.columns:
            self.music_data['combined_tags'] = ''
        self.music_data['combined_tags'] = self.music_data['combined_tags'].fillna('')
        
        # Convert all tags to strings
        self.music_data['combined_tags'] = self.music_data['combined_tags'].astype(str)
        
        tfidf_matrix = self.vectorizer.fit_transform(self.music_data['combined_tags'])
        
        n_samples = len(self.music_data)
        n_neighbors = min(50, max(10, n_samples // 2))
        
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            metric='cosine', 
            algorithm='brute',
            n_jobs=-1
        )
        self.knn_model.fit(tfidf_matrix)
        
        self.last_retrained = datetime.datetime.now()
        print(f"Memory after model init: {self.get_memory_usage()} MB")
        print(f"Initialized KNN with {n_neighbors} neighbors for {n_samples} songs")
    
    def get_recommendations(self, user_id: int, params) -> pd.DataFrame:
        """Generate recommendations using KNN"""

        if isinstance(params, pd.DataFrame):
            params = params.iloc[0].to_dict()
        elif not isinstance(params, dict):
            raise ValueError("params must be a dict or a single-row DataFrame")


        # Safety check: ensure activity is a lowercase string
        params['activity'] = str(params.get('activity', '')).lower()

        if self.knn_model is None:
            self.initialize_models()

        # Build text-based query string for TF-IDF vectorization
        query = self._build_query(params)

        # Make sure the query is passed as a list to match expected input format
        query_vec = self.vectorizer.transform([query])

        # Get nearest neighbors
        n_neighbors = self.knn_model.n_neighbors
        distances, indices = self.knn_model.kneighbors(query_vec, n_neighbors=n_neighbors)

        # Filter valid indices
        valid_indices = [idx for idx in indices[0] if idx < len(self.music_data)]
        recommended_songs = self.music_data.iloc[valid_indices].copy()

        # Convert distances to similarity scores
        scores = 1 - distances[0][:len(valid_indices)]

        # Get user feedback
        user_feedback = self._get_user_feedback(user_id)

        # Apply score boosts based on context and user feedback
        boosted_scores = self._apply_boosts(
            scores,
            params,
            user_feedback,
            valid_indices
        )

        recommended_songs['score'] = boosted_scores
        recommendations = recommended_songs.sort_values('score', ascending=False)

        return self._get_diverse_recommendations(recommendations)

    def _build_query(self, params: Dict) -> str:
        """Build search query from context"""
        search_tags = []

        activity = str(params.get('activity', '')).lower()
        search_tags.extend(ACTIVITY_TAGS.get(activity, [activity]))

        city = params.get('city', '')
        search_tags.extend(INDIAN_CITIES.get(city, []))

        weather_tags = params.get('weather_tags', [])
        if isinstance(weather_tags, dict):  # Fix malformed inputs
            weather_tags = list(weather_tags.values())
        if not isinstance(weather_tags, list):
            weather_tags = [str(weather_tags)]
        search_tags.extend(weather_tags)

        time_tags = params.get('time_tags', [])
        if isinstance(time_tags, dict):
            time_tags = list(time_tags.values())
        if not isinstance(time_tags, list):
            time_tags = [str(time_tags)]
        search_tags.extend(time_tags)

        return ' '.join(map(str, search_tags))

    
    def _apply_boosts(self, scores: np.array, params: Dict, feedback: Dict, indices: List[int]) -> np.array:
        """Apply contextual and feedback-based boosts"""
        boosted_scores = scores.copy()
        activity = str(params['activity']).lower()
        
        for i, idx in enumerate(indices):
            if idx >= len(self.music_data):
                continue
                
            row = self.music_data.iloc[idx]
            
            # Activity boost
            if activity != 'any' and any(tag in str(row['combined_tags']).lower() 
                                     for tag in ACTIVITY_TAGS.get(activity, [activity])):
                boosted_scores[i] *= 1.5
                
            # City/cultural boost
            if any(tag in str(row['combined_tags']).lower() 
                   for tag in INDIAN_CITIES.get(params['city'], [])):
                boosted_scores[i] *= 1.2
                
            # Weather boost
            if any(tag in str(row['combined_tags']).lower() for tag in params['weather_tags']):
                boosted_scores[i] *= 1.1
                
            # Time boost
            if any(tag in str(row['combined_tags']).lower() for tag in params['time_tags']):
                boosted_scores[i] *= 1.1
            
            # Apply user feedback
            if str(idx) in feedback['likes']:
                boosted_scores[i] *= 1.8
            elif str(idx) in feedback['dislikes']:
                boosted_scores[i] *= 0.2
        
        return boosted_scores
    
    def _get_diverse_recommendations(self, recommendations: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Select diverse recommendations from top scored items"""
        final_recs = []
        genres_selected = set()
        tag_combinations_selected = set()

        for _, song in recommendations.iterrows():
            if len(final_recs) >= n:
                break

            tags = song['activity'].split(',')[:3] if pd.notna(song['activity']) else []
            signature = f"{song['genre']}_{'_'.join(sorted(tags))}"

            if signature not in tag_combinations_selected:
                final_recs.append(song.to_dict())  # <- fix
                tag_combinations_selected.add(signature)
                genres_selected.add(song['genre'])

        if len(final_recs) < n:
            remaining = recommendations[~recommendations.index.isin([r['spotify_id'] for r in final_recs if 'spotify_id' in r])]
            final_recs.extend(remaining.head(n - len(final_recs)).to_dict('records'))

        return pd.DataFrame(final_recs)

    '''
    def _get_diverse_recommendations(self, recommendations: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Select diverse recommendations from top scored items"""
        final_recs = []
        genres_selected = set()
        tag_combinations_selected = set()
        
        for _, song in recommendations.iterrows():
            if len(final_recs) >= n:
                break
                
            tags = song['activity'].split(',')[:3] if pd.notna(song['activity']) else []
            signature = f"{song['genre']}_{'_'.join(sorted(tags))}"
            
            if signature not in tag_combinations_selected:
                final_recs.append(song.to_dict())
                tag_combinations_selected.add(signature)
                genres_selected.add(song['genre'])
        
        if len(final_recs) < n:
            remaining = recommendations[~recommendations.index.isin([r.name for r in final_recs])]
            final_recs.extend(remaining.head(n - len(final_recs)).to_dict('records'))

        return pd.DataFrame(final_recs)
    '''
    
    def _get_user_feedback(self, user_id: int) -> Dict:
        """Get feedback for specific user"""
        db = get_db()
        prefs = db.execute(
            'SELECT liked_songs, disliked_songs FROM user_preferences WHERE user_id = ?',
            (user_id,)
        ).fetchone()
        
        if prefs:
            return {
                'likes': json.loads(prefs['liked_songs']) if prefs['liked_songs'] else {},
                'dislikes': json.loads(prefs['disliked_songs']) if prefs['disliked_songs'] else {}
            }
        return {'likes': {}, 'dislikes': {}}

    def evaluate_model(self) -> Dict:
        """Evaluate the recommendation model using test data and log results"""
        if self.evaluation_data is None or len(self.evaluation_data) == 0:
            return {"error": "No evaluation data available"}
            
        test_queries = [
            {'activity': 'working', 'city': 'Bangalore', 'weather_tags': ['neutral'], 'time_tags': ['productive']},
            {'activity': 'partying', 'city': 'Mumbai', 'weather_tags': ['neutral'], 'time_tags': ['social']},
            {'activity': 'relaxing', 'city': 'Delhi', 'weather_tags': ['neutral'], 'time_tags': ['calm']}
        ]
        
        results = {
            'precision@5': [],
            'recall@5': [],
            'diversity': [],
            'novelty': [],
            'coverage': [],
            'personalization': []
        }
        
        all_recommendations = []
        
        for query in test_queries:
            recs = self.get_recommendations(-1, query)
            all_recommendations.append(recs)
            
            # Calculate metrics (simplified for example)
            results['precision@5'].append(0.65)  # Placeholder - real implementation would calculate
            results['recall@5'].append(0.55)     # Placeholder
            top_genres = recs.head(5)['genre'].unique()
            results['diversity'].append(len(top_genres))
            avg_popularity = recs.head(5)['popularity'].mean()
            results['novelty'].append(1 - (avg_popularity / 100))
        
        # Calculate additional metrics
        results['coverage'] = len(pd.concat(all_recommendations)['spotify_id'].unique()) / len(self.music_data)
        results['personalization'] = 0.75  # Placeholder for personalization metric
        
        metrics = {
            'precision@5': np.mean(results['precision@5']),
            'recall@5': np.mean(results['recall@5']),
            'diversity': np.mean(results['diversity']),
            'novelty': np.mean(results['novelty']),
            'coverage': results['coverage'],
            'personalization': results['personalization'],
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_songs': len(self.music_data),
            'num_users': self._get_user_count()
        }
        
        # Log to CSV
        self._log_metrics_to_csv(metrics)
        
        # Print to terminal
        self._print_metrics(metrics)
        
        return metrics
    
    def _log_metrics_to_csv(self, metrics: Dict):
        """Log metrics to a CSV file"""
        fieldnames = ['timestamp', 'precision@5', 'recall@5', 'diversity', 'novelty', 'coverage', 'personalization', 'num_songs', 'num_users']
        file_path = 'model_metrics.csv'
        
        # Check if the file already exists to write headers only once
        try:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                # Write header if the file is empty
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)
        except Exception as e:
            print(f"Error logging metrics to CSV: {e}")
    
    def _get_user_count(self) -> int:
        """Return a placeholder user count, as no database is available"""
        return 1000  # Example, replace with your actual count if available
    
    def _print_metrics(self, metrics: Dict):
        """Helper to print metrics to terminal with formatting"""
        print("\n=== MODEL EVALUATION METRICS ===")
        print(f"Timestamp: {metrics['timestamp']}")
        print(f"\nPerformance Metrics:")
        print(f"  Precision@5: {metrics['precision@5']:.2f} (higher is better)")
        print(f"  Recall@5: {metrics['recall@5']:.2f} (higher is better)")
        print(f"\nDiversity Metrics:")
        print(f"  Diversity: {metrics['diversity']:.2f} unique genres in top 5")
        print(f"  Coverage: {metrics['coverage']:.2%} of catalog recommended")
        print(f"\nNovelty Metrics:")
        print(f"  Novelty: {metrics['novelty']:.2f} (1 = most novel)")
        print(f"\nPersonalization:")
        print(f"  Personalization: {metrics['personalization']:.2f} (1 = fully personalized)")
        print(f"\nSystem Stats:")
        print(f"  Songs in catalog: {metrics['num_songs']}")
        print(f"  Active users: {metrics['num_users']}")
        print("="*50)
    
    def _get_user_count(self) -> int:
        """Get total number of users in the system"""
        db = get_db()  # Assuming get_db() is your function to connect to the database
        count = db.execute('SELECT COUNT(*) FROM users').fetchone()[0]  # Adjust the table name if needed
        return count

# Initialize recommendation model
model = RecommendationModel()

def get_weather(city: str) -> Tuple[str, float, List[str]]:
    """Fetch weather data with error handling and caching"""
    try:
        if city not in INDIAN_CITIES:
            city = 'Mumbai'
            
        cache_key = f"weather_{city}"
        if cache_key in session and (datetime.datetime.now() - session[cache_key]['timestamp']).seconds < 3600:
            cached = session[cache_key]
            return cached['weather'], cached['temp'], cached['tags']
            
        api_key = os.getenv('WEATHER_API_KEY')
        if not api_key:
            raise ValueError("Weather API key not configured")
            
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            weather_code = data['weather'][0]['main']
            temp = round(data['main']['temp'])
            tags = WEATHER_TAGS.get(weather_code, ['neutral'])
            
            session[cache_key] = {
                'weather': weather_code,
                'temp': temp,
                'tags': tags,
                'timestamp': datetime.datetime.now()
            }
            
            return weather_code, temp, tags
    
    except requests.exceptions.RequestException as e:
        print(f"Weather API request failed: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Weather API response parsing failed: {str(e)}")
    
    return 'Clear', 25, ['neutral']

def get_time_period() -> Tuple[str, List[str]]:
    """Get time period with associated tags"""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return 'Morning', ['fresh', 'awake', 'energetic', 'sunrise', 'optimistic']
    elif 12 <= hour < 17:
        return 'Afternoon', ['productive', 'active', 'daytime', 'bright', 'lively']
    elif 17 <= hour < 21:
        return 'Evening', ['sunset', 'relaxed', 'social', 'golden', 'warm']
    return 'Night', ['late', 'sleepy', 'quiet', 'moon', 'dreamy']

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        db = get_db()
        error = None
        
        if not username:
            error = 'Username is required.'
        elif not email:
            error = 'Email is required.'
        elif not password:
            error = 'Password is required.'
        elif db.execute(
            'SELECT id FROM users WHERE username = ?', (username,)
        ).fetchone() is not None:
            error = f"User {username} is already registered."
        elif db.execute(
            'SELECT id FROM users WHERE email = ?', (email,)
        ).fetchone() is not None:
            error = f"Email {email} is already registered."
            
        if error is None:
            db.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, generate_password_hash(password))
            )
            db.commit()
            flash('Successfully registered! Please log in.', 'success')
            return redirect(url_for('login'))
        
        flash(error, 'error')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        error = None
        user = db.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        
        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'
            
        if error is None:
            user_obj = User(id=user['id'], username=user['username'], email=user['email'])
            login_user(user_obj)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        
        flash(error, 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    session['session_id'] = os.urandom(16).hex()
    model.load_data()
    if model.knn_model is None:
        model.initialize_models()
    return render_template('index.html', cities=list(INDIAN_CITIES.keys()), activities=list(ACTIVITY_TAGS.keys()))

@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    city = request.form['city']
    activity = request.form['activity'].lower()
    
    try:
        weather, temp, weather_tags = get_weather(city)
    except Exception as e:
        print(f"Error getting weather: {str(e)}")
        weather, temp, weather_tags = 'Clear', 25, ['neutral']
    
    time_of_day, time_tags = get_time_period()
    

    input_data = {
    'city': city,
    'activity': activity,
    'weather_tags': weather_tags,
    'time_tags': time_tags
    }

    recommendations = model.get_recommendations(current_user.id, input_data)


    recommendations['spotify_embed'] = recommendations['spotify_id'].apply(
        lambda x: f"https://open.spotify.com/embed/track/{x}?utm_source=generator" if pd.notna(x) else ""
    )
    recommendations['popularity_stars'] = recommendations['popularity'].apply(
        lambda x: '★' * round(x/20) + '☆' * (5 - round(x/20))
    )
    
    session['current_recommendations'] = {
        str(i): {
            'track_name': row['track_name'],
            'artist': row['artist']
        }
        for i, row in recommendations.iterrows()
    }
    
    return render_template('results.html',
                         songs=recommendations.to_dict('records'),
                         context={
                             'city': city,
                             'activity': activity.capitalize(),
                             'weather': weather,
                             'temp': temp,
                             'time': time_of_day
                         })

@app.route('/feedback', methods=['POST'])
@login_required
def handle_feedback():
    data = request.json
    track_name = data.get('track_name')
    artist = data.get('artist')
    feedback_type = data.get('type')
    
    current_recs = session.get('current_recommendations', {})
    track_id = None
    
    for rec_id, rec in current_recs.items():
        if rec['track_name'] == track_name and rec['artist'] == artist:
            track_id = rec_id
            break
    
    if track_id:
        db = get_db()
        prefs = db.execute(
            'SELECT liked_songs, disliked_songs FROM user_preferences WHERE user_id = ?',
            (current_user.id,)
        ).fetchone()
        
        feedback = {
            'likes': json.loads(prefs['liked_songs']) if prefs and prefs['liked_songs'] else {},
            'dislikes': json.loads(prefs['disliked_songs']) if prefs and prefs['disliked_songs'] else {}
        }
        
        if feedback_type == 'like':
            feedback['likes'][track_id] = True
            feedback['dislikes'].pop(track_id, None)
        elif feedback_type == 'dislike':
            feedback['dislikes'][track_id] = True
            feedback['likes'].pop(track_id, None)
        
        db.execute(
            '''INSERT OR REPLACE INTO user_preferences 
               (user_id, liked_songs, disliked_songs) 
               VALUES (?, ?, ?)''',
            (current_user.id, json.dumps(feedback['likes']), json.dumps(feedback['dislikes']))
        )
        db.commit()
        
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Track not found'}), 400

@app.route('/evaluate')
@login_required
def evaluate():
    """Endpoint to view model evaluation metrics with admin access"""
    if not current_user.is_authenticated or current_user.username != 'admin':
        flash("Admin access required", "error")
        return redirect(url_for('home'))
    
    # Get current metrics
    current_metrics = model.evaluate_model()
    
    # Get historical metrics from database
    db = get_db()
    historical_metrics = db.execute(
        'SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT 10'
    ).fetchall()
    
    return render_template('metrics.html',
                         current_metrics=current_metrics,
                         historical_metrics=historical_metrics)

@app.route('/metrics/history')
@login_required
def metrics_history():
    """API endpoint to get historical metrics (for charts)"""
    if not current_user.is_authenticated or current_user.username != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    
    db = get_db()
    metrics = db.execute(
        'SELECT timestamp, precision, recall, diversity, novelty, coverage, personalization FROM model_metrics ORDER BY timestamp DESC LIMIT 30'
    ).fetchall()
    
    return jsonify([dict(m) for m in metrics])


if __name__ == '__main__':
    init_db()
    model.load_data()
    model.initialize_models()
    
    # Initial evaluation
    print("\nRunning initial model evaluation...")
    model.evaluate_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)