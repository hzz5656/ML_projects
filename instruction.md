# Twill ML Assignment

Please **DO NOT** share or discuss the assignments.

For any further questions, feel free to contact thomas@twill.health

## Question 2

**Transformer-based Recommendation System**

![Rec-Model](./pics/recmodel.png)


### 1. Data

1. Movie datasets can be found in `./data/ml-1m/`
2. [Dataset README](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt) includes all the necessary information about the dataset.
3. Merge `overview` column from `movies_metadata.csv` into `movies.dat` based on `title` and `release_date` (use as text feature)

### 2. Modeling 
(`Pytorch` is recommended)
1. Please use [`transformers`](https://github.com/huggingface/transformers) library for BERT models.
2. Encode user & item features (convert features into numeric representations)
    - Text Features (BERT embedding layer).
    - Categorical (Sparse) Features (`torch.nn.Embedding`).
    - Continuous (Dense) Features.

        a. Concatenate to Embedding directly. (Figure 4 in W&D [3])

        b. Bucketize and treat them as Categorical Features.

        c. (Optional) Follow up question: is option `a` better or `b`?

4. Feed into 1 transformer layer.
5. BERT as feature extraction pseudocode in Python:
    ```python
    # Pesudocode for `Second-to-Last Hidden` shown in paper

    # return all 12 layers of hidden states [12, bath_size, max_length, 768]
    hidden_states = model(tokens)
    
    # get the last 2 layers and take average along layer-axis
    # [2, batch_size, max_length, 768] -> [batch_size, max_length, 768]
    last_2_layer_hidden_states = torch.mean(hidden_states[-2:], axis=0)
    
    # mean pooling along token-axis [bath_size, max_length, 768] -> [batch_size, 1, 768].
    # each of the sample will have [1, 768] sentence embedding
    sentence_embedding = torch.mean(last_2_layer_hidden_states, axis=1)
    ```
    For more details, please refer to Table 7 in [paper](https://arxiv.org/pdf/1810.04805.pdf)

5. Concat Transformer layer outputs with BERT outputs.
6. 3 x dense layer with ReLU.

### 3. Loss & Evaluation
1. Loss: MSE
2. Metric: MAE

    Feel free to use other metrics.

    
### 5. Documentations
1. Refactor `README.md`
2. Update `requirements.txt`
3. Using Docker is recommended


You **DO NOT** need to fully train the model until or optimize for eval metrics, but **it has be reproducible**.

The idea is to assess:

- coding & modeling skill.
- mainstream frameworks (pytorch, tensorflow, and etc).
- research skill under limited resources.
- reproducibility
- creativity


### Resources:

[1] https://keras.io/examples/structured_data/movielens_recommendations_transformers/

[2] https://arxiv.org/pdf/1905.06874.pdf

[3] https://arxiv.org/pdf/1606.07792.pdf
