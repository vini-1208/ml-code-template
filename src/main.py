##few key points:

# For demo purposes as well as from validaiton standpoint: subset data points
# 1. subset movies (instead of ~7k to reduce sparsity) to a few ~100s (apply random sampling??)
# 2. subset users (~few 100s)
# 3. build additional binary matrix for 1s and Os for movies watched


# Questions
# 1. how to extract genres for movies
# 2. how to dump jsons into Neo4J
# 3. users- movies ratings
# 4. Create graph embeddings with all relationships in embeddings


# from preprocess.feature_embedding import
import os
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import mean_absolute_error

# some constants #move to config
INPUT_PATH = r"C:\Users\vinita.phadke\OneDrive - Fractal Analytics Pvt. Ltd\Documents\Fractal\SKU\Knowledge Graph Applications\recommender system\knowledge_graphs\data"
OUTPUT_PATH = r"C:\Users\vinita.phadke\OneDrive - Fractal Analytics Pvt. Ltd\Documents\Fractal\SKU\Knowledge Graph Applications\recommender system\knowledge_graphs\data\final recommendations"
RECOMM_TYPE = "embedding_multirelation_2"
MOVIES_MAT_COLS = ["n.identity", "n.properties.imdbRating", "n.properties.revenue"]
MOVIES_KEY_ID = "n.identity"
THRESHOLD = 0.4
TOP_N = 1000

## if __name__ == '__main__':
# load movies dataset
movies_df = pd.read_csv(os.path.join(INPUT_PATH, "movies.csv"))

# subset required fields and movies_id
movies_mat_df = movies_df.loc[movies_df["n.properties.languages"] == "['English']"][
    MOVIES_MAT_COLS
]

# subset movies with revenue more than 10k.
# movies reduced from 9125 to 3242
movies_mat_df = movies_mat_df.loc[movies_mat_df["n.properties.revenue"] > 10000]

# check for missing columns and drop those rows
movies_mat_df.isnull().sum()
movies_mat_df.dropna(inplace=True)

# load genres.csv for merging genre attributes
genre_df = pd.read_csv(os.path.join(INPUT_PATH, "genre.csv"))
genre_df = genre_df.loc[genre_df["end.properties.name"] != "(no genres listed)"]
genre_df["flag"] = 1
genre_df = genre_df.pivot(
    index="relationship.start", columns="end.properties.name", values="flag"
)
genre_df = genre_df.fillna(0)
genre_df.reset_index(inplace=True)

# no. of movies reduced to 3209
movies_mat_df = pd.merge(
    movies_mat_df,
    genre_df,
    left_on="n.identity",
    right_on="relationship.start",
    how="inner",
)

##only for temporary subsetting##
movies_mat_df = movies_mat_df

# load user ratings data
user_df = pd.read_csv(os.path.join(INPUT_PATH, "users.csv"))

# subset movies for short demo and validation purposes# #start
udf = user_df.loc[
    user_df["relationship.end"].isin(movies_mat_df["n.identity"].values.tolist())
]
# udf = udf.loc[udf['start.properties.userId'].isin([30,15,19,23,55,67])]

udf_ = pd.pivot_table(
    udf,
    values="relationship.properties.rating",
    index=["start.properties.userId"],
    columns=["relationship.end"],
    aggfunc=np.sum,
)
udf_ = udf_.fillna(0)

##TODO later, intersection between movies matrix and udf is 1573 movies?!
movies_mat_df = movies_mat_df.loc[
    movies_mat_df["n.identity"].isin(udf_.columns.tolist())
]
# subset users and movies for short demo and validation purposes# #end

##for graph embeddings##
if "embedding" in RECOMM_TYPE:
    MOVIES_KEY_ID = "nodeId"
    movie_embeding = pd.read_csv(
        os.path.join(INPUT_PATH, "embeddings_multi_relation.csv")
    )
    movie_embeding = pd.merge(
        movie_embeding,
        movies_mat_df["n.identity"],
        left_on="nodeId",
        right_on="n.identity",
        how="inner",
    )
    movies_mat_df = movie_embeding
    del movies_mat_df["n.identity"]
##graph embeddings end###


# normalize columns
movies_mat_df = movies_mat_df.set_index(MOVIES_KEY_ID)

scaler = MinMaxScaler()
movies_mat_scaled = scaler.fit_transform(movies_mat_df)
# build movie-movie similarity matrix
movies_sim_mat = cosine_similarity(
    movies_mat_scaled
)  ##IMP TODO decide later whether to pass raw matrix or scaled one
movies_sim_mat.min()

# take user-movies ratings matrix as input
# scaler = MinMaxScaler()
# udf_norm = scaler.fit_transform(udf_)
# udf_norm = normalize(udf_.select_dtypes(exclude=[object]), axis=0)


# item based collaborative filtering begins#

# threshold movies based on cosine scores

sim_matrix = movies_sim_mat
sim_matrix1 = sim_matrix > THRESHOLD
sim_matrix = np.multiply(sim_matrix1, sim_matrix)


# select top-N for computation and remaining as false/ ==0
def find_topn(input_array, N=20):
    a = np.argsort(input_array)
    input_array[a[: a.shape[0] - N]] = 0
    return input_array


sim_filt = sim_matrix.copy()
sim_filt = np.apply_along_axis(find_topn, 1, sim_filt.copy(), N=TOP_N + 1)
np.fill_diagonal(
    sim_filt, 0
)  # we do not want self ratings to be considered while computing scores - as MAE is calculated on true ratings, we want predictions based only on similar movies

# dot product functions in two parts:
# predict(rat_matrix.values,rating_method,sim_filt)

rating_mat = udf_.values
rating_method = "weighted_avg"
rating_bool_mat = rating_mat > 0
rating_mat = np.multiply(
    rating_mat, rating_bool_mat
)  # ensuring our rating matrix is non negative
if rating_method == "weighted_avg":
    numerator = np.dot(rating_mat, sim_filt)
    denominator = np.dot(rating_bool_mat, sim_filt)
    prediction = numerator / denominator
    prediction = np.nan_to_num(prediction)

# get prediction ratings corresponding to true ratings
prediction_bool_mat = np.multiply(prediction, rating_bool_mat)

# compute MAE
def mae(prediction, ground_truth):
    predict_list = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    # logger.debug(ground_truth.shape)
    print(ground_truth.shape)
    # logger.debug(predict_list.shape)
    print(predict_list.shape)
    return mean_absolute_error(ground_truth, predict_list)


# compute mae
mae(prediction_bool_mat, rating_mat)

# steps to create top-n ranking for every user
# keep only those predictions which have not been watched/rated by user
rating_bool_unrated = rating_mat <= 0
prediction_mat_unrated = np.multiply(prediction, rating_bool_unrated)

# identify top-n recommendations for every user per row
topN = 15
topn_mat1 = np.apply_along_axis(find_topn, 1, prediction_mat_unrated.copy(), N=topN)

pd.DataFrame((topn_mat1 != 0).sum(1))
pd.DataFrame((topn_mat1 != 0).sum(1)).shape
# apply sorting logic to get column names i.e movie IDs
topn_mat1 = pd.DataFrame(topn_mat1, columns=udf_.columns.tolist())
topn_mat1 = pd.DataFrame(topn_mat1.columns[np.argsort(-topn_mat1.values, axis=1)])
topn_mat2 = topn_mat1.iloc[:, :topN]

# fetch movie names corresponding to movie IDs:
top_recom = topn_mat2.values

movies_dict = dict(zip(movies_df["n.identity"], movies_df["n.properties.title"]))

final_recom = [movies_dict[movie] for movie in top_recom.flatten()]

# np.array(final_recom).reshape(top_recom.shape[0],top_recom.shape[1])
final_recom_df = pd.DataFrame(
    np.array(final_recom).reshape(top_recom.shape[0], top_recom.shape[1]),
    index=udf_.index,
)
final_recom_df.to_csv(os.path.join(OUTPUT_PATH, RECOMM_TYPE + "_recommendations.csv"))


# Compute MAP@k on ground truth vs prediction
# create matrix with top-10 movies highly rated movies (in decreasing order) for both actual and predicted matrix

topN_actual = 15
actual_recom = np.apply_along_axis(find_topn, 1, rating_mat.copy(), N=topN_actual)
actual_recom = pd.DataFrame(actual_recom, columns=udf_.columns.tolist())
actual_recom = pd.DataFrame(
    actual_recom.columns[np.argsort(-actual_recom.values, axis=1)]
)
actual_recom = actual_recom.iloc[:, :topN_actual]
actual_recom_mat = actual_recom.values
actual_recom_mat.sort(axis=1)

##prediction matrix
pred_recom_gr_truth = np.apply_along_axis(
    find_topn, 1, prediction_bool_mat.copy(), N=topN_actual
)  # identify topN movies based on ratings and make other ratings as 0
pred_recom_gr_truth = pd.DataFrame(
    pred_recom_gr_truth, columns=udf_.columns.tolist()
)  # convert to df
pred_recom_gr_truth = pd.DataFrame(
    pred_recom_gr_truth.columns[np.argsort(-pred_recom_gr_truth.values, axis=1)]
)  # sort df in decreasing order of ratings and fill cel values coressponding to column names
pred_recom_gr_truth = pred_recom_gr_truth.iloc[:, :topN_actual]  # select topN only
pred_recom_gr_truth_mat = pred_recom_gr_truth.values
pred_recom_gr_truth_mat.sort(axis=1)
# for every item from 1 through k, calculate mean


act1 = actual_recom_mat
pred1 = pred_recom_gr_truth_mat


mean_recall = 0.0
for index in range(act1.shape[0]):
    print(act1[index])
    print(pred1[index])
    print(np.intersect1d(act1[index], pred1[index]))
    print(len(np.intersect1d(act1[index], pred1[index])))
    mean_recall = mean_recall + (
        len(np.intersect1d(act1[index], pred1[index])) / (len(act1[index]))
    )

mean_recall = mean_recall / act1.shape[0]
