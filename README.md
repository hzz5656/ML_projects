# Movie Recommender System

This project is a simplified version of a movie recommender system based on the IMDB movie dataset. The original project cannot be shared due to a confidential agreement, but this serves as a reference with a similar approach.

## Overview

The Movie Recommender System aims to provide personalized movie recommendations to users based on their preferences. The system utilizes the IMDB movie dataset to gather information about movies, including genres, ratings, and other relevant features.

## Approach

The recommender system follows a simplified approach, but the general idea is similar to the original project. The key steps include:

Data Loading: The IMDB movie dataset is used as the source of information. The dataset includes details such as movie titles, genres, ratings, and other metadata.

Data Preprocessing: The data is cleaned and preprocessed to handle missing values, remove duplicates, and ensure consistency. Features relevant to the recommendation task are selected for further analysis.

Content-Based Filtering: The system uses a content-based filtering approach to recommend movies. This method recommends items based on the user's past preferences and the characteristics of the items. In this case, movie genres, ratings, and other features are used to determine similarity between movies.

User Input: Users can input their preferences, such as favorite genres or past-rated movies. The system takes this input into account when generating personalized recommendations.

Recommendation Generation: Based on the user input and content-based filtering, the system generates a list of recommended movies for the user. The recommendations are sorted based on their similarity to the user's preferences.

## System Requirements

Please note that the transformer layer in this demo might not have run completely due to limited RAM on the author's personal PC. The code, however, is designed to be runnable on a system with better RAM and GPU capacity.

### Author's System Configuration
Processor: Intel i5 8th Gen
RAM: 16 GB
GPU: NVIDIA GeForce GTX 1060

For optimal performance and to ensure the complete execution of the transformer layer, it is recommended to use a PC with the following specifications:

Processor: Intel i7 or equivalent
RAM: 32 GB or higher
GPU: NVIDIA GeForce GTX 3060 or higher (or an equivalent AMD GPU)

The transformer layer, being computationally intensive, benefits significantly from GPU acceleration. Users are advised to consider a system with a dedicated GPU to enhance the speed and efficiency of the recommendation system.

### Running the Code on a System with Higher RAM and GPU
If you encounter memory issues while running the notebook on your personal PC, consider using a PC with higher RAM and a dedicated GPU. Ensure that you have the necessary GPU drivers and dependencies installed to leverage the GPU for accelerated computations.

Note: Cloud platforms with GPU support, such as Google Colab or AWS Sagemaker, can be utilized for running the code if a local system with high RAM and GPU is not available. Adjust the code to enable GPU acceleration as needed.



## Usage

To run the recommender system:

Clone the repository to your local machine.

Open the main.ipynb notebook using Jupyter or any compatible environment.

Follow the instructions within the notebook to load the data, preprocess it, and generate movie recommendations.

Note: This project is a simplified version for reference purposes only. The original project, which is confidential, may have additional features, optimizations, and a more sophisticated algorithm.
