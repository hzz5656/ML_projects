# Overview

Welcome to my movie recommendation system project! This project is centered around building a recommendation model, mainly following the instructions outlined in `instruction.md`.

## Modeling Approach

### Continuous Features Transformation
For continuous features like age, the input table has already transformed them into categorical features by creating age groups. 

### Handling Question #2
In response to "### 2. Modeling question #2," I chose to bucketize continuous features and treat them as categorical. Given the absence of explicit numeric or continuous features, bucketizing provided a suitable solution.

## Model Complexity and Input Size

The input table for the recommendation model is extensive, and the embedding process is intricate. Due to the large dataset, I implemented data batching using PyTorch's DataLoader to manage memory usage efficiently.

## BERT Embedding and RAM Constraints

Despite employing data batching techniques, running the final version of the model became challenging. My machine's RAM was heavily utilized during the BERT embedding process, preventing the execution of the final dense layer and evaluation steps.

## Note

Please note that due to resource constraints, the last dense layer and evaluation components could not be executed on my machine. While I incorporated various optimizations, running the final model may require additional resources or cloud computing.

Feel free to explore the provided code, and I welcome any suggestions or contributions to enhance the model's efficiency and performance.