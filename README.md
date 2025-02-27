# Polars Vector Store

A few days ago, I came across this nice blog post: https://minimaxir.com/2025/02/embeddings-parquet/. The author argues that
one of the best ways to store embeddings is to use parquet files. That's a nice idea, also because we have great libraries that help us deal with Parquet files, like Polars.

I agree with him, that's what we have been doing in the ML cube Platform for a while now. 

But I figured: why limit ourselves to just storing embeddings in Parquet files? We can do directly use Polars as a vector store, asking it to store and later retrieve the vector closest to a given query vector, doing all operations in Polars (without extracting the embedding column to a numpy array, similar to what the author of the blog post does).

An additional advantage is that we can also leverage "classical" dataframe operations to add any kind of metadata to the vectors and then query them using these metadata as well.