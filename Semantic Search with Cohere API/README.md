<h1 align="center">Semantic Search with Cohere API</h1>

## Project Aim:
To develop a semantic search tool using Cohere API.

## Project Outline:
In this project I have built a Seamntic Search tool which is trained on a set of data containing questions. This semantic tool is similar to a search suggestions made by Google search engine based on the search query made by a user.

The approach to the problem is tackled in the below steps:

1. Get the dataset of questions.
2. Create Embeddings and index of Embeddings for the dataset.
3. Search using an index and nearest neighbor search.
4. Visualize the archive based on the embeddings.

## Dataset:

[Trec](https://www.tensorflow.org/datasets/catalog/trec) is used in this project, which contains questions and their categories.

## Result:
Sample Result
<p><img
  src=""
  alt="Sample Result"
  title="Sample Result"
  style="display: inline-block; margin: 0 auto; width:500px; height:300px"/></p>
<b>Faces detected with keypoints marked on each of the detected face</b> <br>
 

## Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [Cohere API and API Key](https://os.cohere.ai)
- UMAP
- Altair
- AnnoyIndex

