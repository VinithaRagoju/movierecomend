from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
# Load your movie ratings dataset (assuming 'userId', 'movieId', 'rating' columns)
ratings = pd.read_csv('ratings.csv')
# Create a Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
# Use user-based collaborative filtering with k-NN
sim_options = {  
   'name': 'cosine', 
   'user_based': True
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)
# Make predictions on the test set
predictions = model.test(testset)
# Evaluate the model
accuracy.rmse(predictions)
# Function to get movie recommendations for a user
def get_movie_recommendations(user_id, model, n=5):   
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()  
    movies_to_predict = [movie_id for movie_id in ratings['movieId'].unique() if movie_id not in user_movies]
    predicted_ratings = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movies_to_predict]    
    top_recommendations = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:n]
    return top_recommendations
# Get movie recommendations for a user 
user_id_to_recommend = 1  # Change to the desired user
recommendations = get_movie_recommendations(user_id_to_recommend, model, n=5)
print(f"Top 5 movie recommendations for User {user_id_to_recommend}: {recommendations}")
