"""
cf_main.py implements collaborative filtering framework 
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from models.train import CFTraining
from models.predict import CFPrediction
from models.eval import (
    compute_mae,
    generate_ranking,
    generate_topn_eval,
    compute_mean_recall,
)
from utils.common_utils import logging_creation, timer


if __name__ == "__main__":
    # load config files
    CONFIG_PATH = "../config/cf_config.yaml"
    config = yaml.load(open(CONFIG_PATH), Loader=yaml.FullLoader)
    MOVIES_KEY_ID = config["MOVIES_KEY_ID"]
    FEATURE_TYPE = config["FEATURE_TYPE"]
    MODEL_VERSION = config["MODEL_VERSION"]
    MODEL_PARAMS = config["MODEL_PARAMS"]
    # initialize logging class
    logger = logging_creation(
        logger_name=os.path.join(config["MODEL_LOGPATH"], "cf_experiments")
    )
    logger.info("Load movies dataset and subset English movies")
    # load movies dataset
    movies_df = pd.read_csv(os.path.join(config["INPUT_PATH"], "movies.csv"))
    # subset required fields and movies_id
    movies_mat_df = movies_df.loc[movies_df["n.properties.languages"] == "['English']"][
        config["MOVIES_MAT_COLS"]
    ]
    # subset movies with revenue more than 10k
    movies_mat_df = movies_mat_df.loc[movies_mat_df["n.properties.revenue"] > 10000]
    # check for missing columns and drop those rows
    movies_mat_df.isnull().sum()
    # logger.info('Drop rows with missing columns')
    movies_mat_df.dropna(inplace=True)
    # load genres.csv for merging genre attributes
    logger.info("Load genres for merging genre attributes")
    genre_df = pd.read_csv(os.path.join(config["INPUT_PATH"], "genre.csv"))
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
        genre_df.rename(columns={"relationship.start": "n.identity"}),
        on="n.identity",
        how="inner",
    )

    # load user ratings data
    logger.info("Load user rating data")
    rating_df = pd.read_csv(os.path.join(config["INPUT_PATH"], "users.csv"))

    # subset movies which are available in common between user and movie matrix
    rating_df = rating_df.loc[
        rating_df["relationship.end"].isin(movies_mat_df["n.identity"].values.tolist())
    ]

    rating_df = pd.pivot_table(
        rating_df,
        values="relationship.properties.rating",
        index=["start.properties.userId"],
        columns=["relationship.end"],
        aggfunc=np.sum,
    )
    rating_df = rating_df.fillna(0)


    ##TODO quality checks later, intersection between movies matrix and rating_df is 1573 movies?!
    logger.info(
        "Take intersection between user and movies matrix so we have same movies in both matrices"
    )
    movies_mat_df = movies_mat_df.loc[
        movies_mat_df["n.identity"].isin(rating_df.columns.tolist())
    ]
    # subset users and movies for short demo and validation purposes# #end

    ##for graph embeddings##
    if config["FEATURE_TYPE"] in ["embedding", "combined"]:
        logger.info("Load embeddings file")
        movie_embeding = pd.read_csv(
            os.path.join(config["INPUT_PATH"], config["INPUT_FILE_EMBEDDINGS"])
        )
        if "embedding" in config["FEATURE_TYPE"]:
            COLS_TO_KEEP = ["n.identity"]  # keep only key column
        else:
            COLS_TO_KEEP = movies_mat_df.columns.tolist()  # keep other features as well
        movies_mat_df = pd.merge(
            movies_mat_df[COLS_TO_KEEP],
            movie_embeding.rename(columns={"nodeId": "n.identity"}),
            on="n.identity",
            how="inner",
        )
    ##graph embeddings end###

    movies_mat_df = movies_mat_df.set_index(MOVIES_KEY_ID)
    logger.info("List of columns for Movies matrix:")
    logger.info(movies_mat_df.columns.tolist)
    # normalize columns
    logger.info("Scaling movie feature matrix..")
    scaler = MinMaxScaler()
    movies_mat_scaled = scaler.fit_transform(movies_mat_df)
    # build movie-movie similarity matrix
    logger.info("Building movie similarity matrix")
    movies_sim_mat = cosine_similarity(movies_mat_scaled)
    logger.info("minimum value of similarity matrix")
    logger.info(movies_sim_mat.min())

    # item based collaborative filtering begins#
    logger.info("Item based collaborative filtering starts..")
    logger.info( "CF Training starts - recompute sim matrix based on threshold and # neighbors")

    trainer = CFTraining(
        model_version=FEATURE_TYPE + "_" + MODEL_VERSION,
        X=movies_sim_mat,
        model_params={
            "sim_threshold": MODEL_PARAMS["sim_threshold"],
            "num_neighbors": MODEL_PARAMS["num_neighbors"],
        },
        logger=logger,
    )

    movies_sim_mat = trainer.fit()
    rating_mat = rating_df.values
    logger.info("CF Prediction starts - compute prediction rating matrix..")
    scorer = CFPrediction(
        model_version=FEATURE_TYPE + MODEL_VERSION,
        X=[rating_mat, movies_sim_mat],
        logger=logger,
    )

    prediction_mat, prediction_eval_mat, prediction_ranking_mat = scorer.predict(
        rating_method=MODEL_PARAMS["rating_method"]
    )

    logger.info("Compute MAE for prediction and actual rating matrix")

    mae_score = compute_mae(
        actual=rating_mat[rating_mat.nonzero()].flatten(),
        predicted=prediction_mat[rating_mat.nonzero()].flatten(),
    )

    logger.info("Generate top-N recommendations for every user..")
    final_recom = generate_ranking(
        rating_df,
        prediction_ranking_mat,
        movies_df,
        movie_key_col="n.identity",
        movie_title_col="n.properties.title",
        top_n=config["TOPN_RANKING"],
    )

    final_recom.to_csv(
        os.path.join(
            config["OUTPUT_PATH"],
            FEATURE_TYPE + "_" + MODEL_VERSION + "_recommendations.csv",
        )
    )

    pd.DataFrame(
        prediction_mat, columns=rating_df.columns.tolist(), index=rating_df.index
    ).to_csv(
        os.path.join(
            config["OUTPUT_PATH"],
            FEATURE_TYPE + "_" + MODEL_VERSION + "_predicted_matrix.csv",
        )
    )

    logger.info("Compute mean recall@k")

    # Compute mean recall@k on ground truth vs prediction
    # i.e For every user, compute recall i.e (# of relevant movies that were recommended / # of top-n relevant recommendations)

    # Step1 For every user, find top-n Movie IDs (in decreasing order of ratings) for both actual and prediction matrix
    actual_recom_eval = generate_topn_eval(rating_df, top_n=config["TOPN_RANKING"])

    prediction_ranking_df = pd.DataFrame(
        prediction_eval_mat, columns=rating_df.columns.tolist(), index=rating_df.index
    )

    pred_recom_eval = generate_topn_eval(
        prediction_ranking_df, top_n=config["TOPN_RANKING"]
    )
    # Step2. compute mean recall
    mean_recall = compute_mean_recall(actual=actual_recom_eval, predicted=pred_recom_eval)

    eval_df = pd.DataFrame(
        [[FEATURE_TYPE + "_" + MODEL_VERSION, mae_score, mean_recall]],
        columns=["Model Version", "MAE", "Mean Recall"],
    )

    eval_df.to_csv(
        os.path.join(
            config["MODEL_EVAL_PATH"],
            FEATURE_TYPE + "_" + MODEL_VERSION + "_eval_metrics.csv",
        ),
        index=False,
    )
