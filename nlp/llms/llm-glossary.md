# [Glossary of LLM Terms - Vectara](https://vectara.com/glossary-of-llm-terms/)
Introduction
--------------------------


The world and promise of Artificial Intelligence, namely Large Language Models, brings excitement and possibility. It also brings a new vernacular that one must understand.

We’ve compiled a glossary of terms focused on Large Language Models (LLMs), like ChatGPT, to help you quickly master machine learning concepts to accelerate your learning and the adoption of AI in your business.

Table of Contents
-----------------

*   [Agentic RAG](https://vectara.com/glossary-of-llm-terms/#h-agentic-rag)
*   [Attention](https://vectara.com/glossary-of-llm-terms/#h-attention)
*   [ChatGPT](https://vectara.com/glossary-of-llm-terms/#h-chatgpt)
*   [Chunking](https://vectara.com/glossary-of-llm-terms/#h-chunking)
*   [Embeddings](https://vectara.com/glossary-of-llm-terms/#h-embeddings)
*   [Few Shot Learning](https://vectara.com/glossary-of-llm-terms/#h-few-shot-learning)
*   [Fine-tuning](https://vectara.com/glossary-of-llm-terms/#h-fine-tuning)
*   [GPTs](https://vectara.com/glossary-of-llm-terms/#h-gpts)
*   [Hallucinations](https://vectara.com/glossary-of-llm-terms/#h-hallucinations-aka-llm-hallucinations)
*   [Hybrid Search](https://vectara.com/glossary-of-llm-terms/#h-hybrid-search)
*   [Inference](https://vectara.com/glossary-of-llm-terms/#h-inference)
*   [Large Language Model (LLM)](https://vectara.com/glossary-of-llm-terms/#h-large-language-model-llm)
*   [Max Marginal Relevance (MMR)](https://vectara.com/glossary-of-llm-terms/#h-mmr)
*   [Mamba](https://vectara.com/glossary-of-llm-terms/#h-mamba)
*   [Neural Search](https://vectara.com/glossary-of-llm-terms/#h-semantic-search-and-neural-search)
*   [Open Source LLM](https://vectara.com/glossary-of-llm-terms/#h-open-source-llm)
*   [Prompt Engineering](https://vectara.com/glossary-of-llm-terms/#h-prompt-engineering)
*   [Reranking](https://vectara.com/glossary-of-llm-terms/#h-reranking)
*   [Retrieval Augmented Generation (RAG)](https://vectara.com/glossary-of-llm-terms/#h-retrieval-augmented-generation-rag)
*   [Semantic Search](https://vectara.com/glossary-of-llm-terms/#h-semantic-search-and-neural-search)
*   [Tokenization](https://vectara.com/glossary-of-llm-terms/#h-tokenization)
*   [Transformers](https://vectara.com/glossary-of-llm-terms/#h-transformers)
*   [Transfer Learning](https://vectara.com/glossary-of-llm-terms/#h-transfer-learning)
*   [Vector Store/Vector Database](https://vectara.com/glossary-of-llm-terms/#h-vector-store-vector-database)

### Agentic RAG

Agentic RAG is an approach to creating AI autonomous agents, using LLMs while incorporating complex reasoning, multi-step planning, function calling and strategic tool use.

The core advantage of Agentic RAG lies in its ability to call tools for information retrieval and task execution purposes. For example, if you ask an Agentic RAG agent to compare the revenue of Apple and Microsoft in 2022, it might analyze the query and call a tool for financial income statement twice – once for Apple and once for Microsoft. Then it will merge the results of both calls to produce the requested response. Agentic RAG can go even beyond this by enabling the model to autonomously analyze the retrieved information execute tasks based on this data.

### Attention

Attention mechanisms in large language models (LLMs) are fundamental components that allow these models to process and understand text more effectively. Introduced by Vaswani et al. in the paper “[Attention is All You Need](https://arxiv.org/abs/1706.03762),” the attention mechanism enables the model to focus on different parts of the input sequence dynamically. Instead of processing the input text sequentially or in fixed chunks, attention allows the model to weigh the importance of each token (word or subword) relative to others in the sequence. This means the model can consider the context of each word in relation to all other words, capturing nuances and dependencies that might be far apart in the text.

The key innovation in the attention mechanism is the self-attention or scaled dot-product attention, where the model computes attention scores between each pair of tokens in the input sequence. These scores determine how much focus to allocate to each token when generating the output. For instance, in a sentence like “The cat sat on the mat,” the model can learn that “cat” and “sat” are closely related, whereas “the” and “on” might be less critical in that specific context.

Attention mechanisms have been instrumental in advancing the state-of-the-art in natural language processing (NLP), powering applications such as machine translation and chatbots, by allowing models to comprehend and generate human-like text with remarkable proficiency.

### ChatGPT

ChatGPT is a conversational chatbot developed by OpenAI. It is based on the GPT (Generative Pre-trained Transformer) architecture, which is a type of deep learning model designed to understand and generate human-like text. The “GPT” in ChatGPT stands for “Generative Pre-trained Transformer,” emphasizing its ability to produce coherent and contextually relevant text based on the prompts it receives.

The underlying technology behind ChatGPT is a [transformer](https://vectara.com/glossary-of-llm-terms/#h-transformers) neural network, which is characterized by its self-attention mechanism.

One of the standout features of ChatGPT is its ability to engage in open-ended conversations with users. Unlike many other chatbots that operate based on predefined rules or scripts, ChatGPT dynamically generates its responses based on the input it receives. This makes it versatile and applicable in a variety of settings, from simple Q&A sessions to more complex conversational interactions.

However, it’s worth noting that while ChatGPT is impressive in its capabilities, it’s not infallible. It relies on the data it was trained on and, as such, can sometimes produce answers that might be off-mark or not entirely accurate – this is also known as [Hallucination](https://vectara.com/glossary-of-llm-terms/#h-hallucinations-aka-llm-hallucinations). Approached like retrieval-augmented-generation enable adapting ChatGPT technology to your own data.

### Chunking

Chunking is a process used to enhance the efficiency and accuracy of information retrieval in natural language processing (NLP) tasks. In RAG, the input text is broken down into smaller, manageable units called “chunks.” These chunks can be sentences, paragraphs, or some other application specific breakdown of a larger text into smaller units.

The purpose of chunking is to improve the retrieval process by allowing the model to focus on more relevant and precise segments of text rather than processing entire documents at once. This approach helps in three ways:

*   **Efficiency:** By working with smaller chunks, the retrieval process becomes faster and requires less computational power.
*   **Accuracy:** Chunks provide more targeted information, reducing noise and improving the relevance of the retrieved data.
*   **Scalability:** Chunking enables the system to handle larger documents more effectively, making it easier to scale the RAG model to accommodate vast amounts of informations.

In RAG, after chunking, each chunk is indexed and stored in a retrieval system (as text and as an embedding vector). When a user issues a query, the chunks most relevant to the query are retreived and used to generate a coherent and informative response.

### Embeddings

In the context of [Large Language Models](https://vectara.com/glossary-of-llm-terms/#h-large-language-model-llm), vector embeddings are a type of representation that captures the semantic meaning or context of words or sentences in a compact form. They are essentially vectors of real numbers (floats), where each dimension could represent a different feature that captures something about that concept’s meaning.

Vector embeddings are also known as “dense representations” whereby continuous vector spaces are used to represent queries and documents, in contrast with the more traditional “sparse representation” whereby each word or phrase is represented as part of the vector, and matching is performed based upon matching of those keywords or phrases between the query and document.

Word embeddings are embedding vectors that map each word to a vector embeddings, whereas sentence embeddings map a complete sentence, paragraph or any text chunk into a vector embedding.

Here’s an example of a sentence embedding:

![Embeddings-vector-and-chunking-example](https://vectara.com/wp-content/uploads/2023/11/Embeddings-vector-and-chunking-example.png)

The chunk of text in red on the left is represented by the vector of real numbers shown on the right.

Vector embeddings serve as input to various NLP tasks such as text classification, sentiment analysis or machine translation. In retrieval-augmented generation applications, sentence embedding models play a crucial role, allowing to match relevant facts from data with user queries.

### Few Shot Learning

In the context of [large language models](https://vectara.com/glossary-of-llm-terms/#h-large-language-model-llm) (LLMS) the concept of few shot learning refers to the ability to provide to the LLM examples of the task you want it to perform, and in this way enhance its performance.

For example, you can ask an LLM to do the following:

“Translate the following from English to French: How are you today?”

In this example there are no examples of the translation so thus would be called “zero shot”.

On the other hand, we can provide the following prompt to the LLM:

“Translate the following from English to French:

1\. Hello → Bonjour

2\. Goodbye → Au revoir

Translate: how are you today?”

In this case we provide two examples of good translation, helping the model do better on the task we want it to excel at.

### Fine-tuning

Fine-tuning is one way to perform [transfer learning](https://vectara.com/glossary-of-llm-terms/#h-transfer-learning). With fine-tuning, you take a pre-trained model and continue training it on a specific (usually smaller, more limited) dataset. There are a few varieties that are common ways to perform fine-tuning, for example:

*   Continue training of complete NN on a specific dataset
*   Freeze some layers and continue training the other layers
*   Add a new layer for a new type of task and only train that layer

Regardless of which specific technique you use for fine-tuning, it’s a much less complicated and significantly less expensive task than full training of an LLM.

Recently techniques like [LORA](https://arxiv.org/abs/2106.09685) (low-rank-adaptation) were suggested for fine-tuning which make it even faster and less expensive, while keeping the performance of fine-tuning similar.

### GPTs

In November of 2023, OpenAI introduced the ability of users or developers to create custom versions of ChatGPT. This ability is aimed at relatively small-scale example, enabling the custom GPT to be configured with a certain tone of voice, to look at the web for knowledge as well as up to 20 PDF documents.

The overall vision is for an “app store” of GPTs, built by developers, that will be available as private GPTs and for everyone to use.

GPTs are best for small-scale customizations of GPT and are not the best solution for enterprise use cases with thousands or millions of documents, where scalable [RAG](https://vectara.com/glossary-of-llm-terms/#h-retrieval-augmented-generation-rag) solutions remain the recommended approach.

### Hallucinations (AKA, LLM Hallucinations)

When an LLM generates text that is nonsensical or unfaithful to the provided source content, we often say that content is an LLM [hallucination](https://vectara.com/glossary-of-llm-terms/#h-hallucinations-aka-llm-hallucinations) (also called LLM fabrication and a variety of similar terms).

As an example, consider the following question to ChatGPT: “Did Will Smith ever hit anyone?”. ChatGPT will respond with (as of July 2023): “There is no public record or credible evidence to suggest that Will Smith has ever physically assaulted or hit anyone. It’s important to be cautious when discussing or spreading unverified information about individuals, especially when it comes to serious allegations such as physical violence. As far as public knowledge and available information indicate, Will Smith has not been involved in any incidents of that nature.”

Clearly, this is wrong since we know of the famous incident of Will Smith hitting Chris Rock at the 2022 Oscars.

LLM Hallucinations occur for two main reasons:

1.  You ask a question to which the answer is NOT “known” to the LLM (i.e. is not available in its training data). In this case, an LLM might caveat its response or refuse to answer, but more likely it will respond with a wrong answer.
2.  An answer to your question is “known” to the LLM, but includes fictional content or content that is subjective like opinions and beliefs.

Grounded Generation, Vectara’s name for Retrieval Augmented Generation, is an approach that provides the LLM with additional, contextual information to avoid hallucinations, and works as follows.

As a first step, user data is provided as additional contextual information. This can be text data from a database, a set of PDF files, or any other data. The data is processed, encoded, and stored within the Vectara platform’s vector store:

![Hallucinations-workflow-1](https://vectara.com/wp-content/uploads/2023/11/Hallucinations-workflow-1.png)

When a user issues a query, that query is encoded to an embedding form, and the most relevant document segments or facts are extracted from the vector store. Then an LLM is used to respond to the user query combining the existing LLM knowledge with the provided facts:

![Hallucinations-workflow-2](https://vectara.com/wp-content/uploads/2023/11/Hallucinations-workflow-2.png)

### HHEM

HHEM, or the Huges Hallucination Evaluation Model is a an open source classification model, [available](https://huggingface.co/vectara/hallucination_evaluation_model) on HuggingFace, that can be used to detect hallucinations. It is particularly useful in the context of building retrieval-augmented-generation (RAG) applications where a set of facts is summarized by an LLM, but the model can also be used in other contexts.

HHEM is also used to rank LLMs by their overall likelihood to hallucinate in this manner, as shown in this [leaderboard](https://huggingface.co/spaces/vectara/leaderboard).

### Hybrid Search

Semantic search provides a fantastic approach to retrieve relevant search results, but it’s not perfect. There are some cases, especially for single-word queries that seek information about a specific product or item, where traditional keyword-based search works better.

Hybrid search attempts to blend semantic or neural search with the traditional keyword-based search, leveraging the precision of keyword search with the contextual understanding of semantic search. By combining both methods, hybrid search offers a more nuanced and effective approach to information retrieval, capable of providing better and more accurate results than either method can offer individually.

### Inference

Inference is the operation by which you generate text with a generative LLM. The input is a trained model and a sequence of tokens, and the output is the next predicted token. If you want to generate a whole sequence with inference, you just generate one token at a time.

Inference with an LLM is a stochastic operation and may not generate the same result every time. Specifically, there are a few parameters that control inference:

1.  **Temperature:** the temperature controls how randomized the response is. If temp=0, then the response is deterministic (always choose the token with highest probability) whereas higher values result in the more randomized results.
2.  **Top\_p and top\_k** are two mechanisms to pick the next token.
    1.  Top\_k is an integer value (e.g. 40 or 100) and determines the length of top tokens to consider. For example if the vocabulary has 50K tokens, and top\_k=50, when selecting the next token we pick the 50 highest likelihood tokens and ignore the rest, then sample from those available tokens
    2.  Top\_p works similarly to top\_k but instead of selecting a certain number of tokens, it picks the top N tokens such that their accumulated probability is equal or higher than the value of top\_p (0…1)

With LLMs that are have a lot of parameters (like Llama2-70b) inference becomes a non-trivial operation. This resulted in many vendors like [Together.AI](https://www.together.ai/), [AnyScale](https://www.anyscale.com/endpoints), [Nvidia](https://catalog.ngc.nvidia.com/ai-foundation-models), [Huggingface](https://huggingface.co/models) and others providing an “inference endpoint” for open source models.

### Large Language Model (LLM)

A Large Language Model (LLM) is a neural network that is trained to understand language and can be used for various tasks, like summarizing, translating, predicting, and generating text. An LLM is trained on massive text datasets such as Common Crawl, WebText2, Books1, Books2, and Wikipedia.

Mathematically, the LLM is learning the probability distribution of text as follows:

**Equation 1:** ![Probability-distribution-of-text](https://vectara.com/wp-content/uploads/2023/11/Probability-distribution-of-text.png)

Where Ti represents the i-th token or word in the text.

A common type of LLM, also called **auto-regressive (AR) LLM**, is one where the model is trained to predict the _next-token,_ conditioned on all previous tokens. GPT-3 and GPT-4 fall into this category.

With AR LLMs, the model is trained to predict the following:

**Equation 2:** ![Auto-regressive-probability-distribution-of-text](https://vectara.com/wp-content/uploads/2023/11/Auto-regressive-probability-distribution-of-text.png)

These are also called generative LLMs.

### MAMBA

[Mamba](https://arxiv.org/pdf/2312.00752) is a novel sequence modeling architecture that offers a promising alternative to Transformer models. It utilizes selective state spaces to achieve linear-time sequence modeling, providing significant improvements in efficiency and scalability compared to the quadratic complexity of Transformers. Mamba’s key innovation lies in its ability to selectively retain or forget information through a selection mechanism, allowing it to handle long-range dependencies effectively.

The Mamba architecture demonstrates impressive performance across various tasks, including language modeling, DNA sequence analysis, and audio generation. It outperforms Transformers of similar size in language modeling tasks and shows the ability to handle context lengths up to 1 million tokens with linear scaling. Notably, Mamba offers 5x higher generation throughput compared to Transformers of similar size and can match the performance of Transformers twice its size in both pre-training and downstream evaluation.

### Max Marginal Relevance

In the context of retrieval augmented generation (RAG), [MMR](https://vectara.com/blog/get-diverse-results-and-comprehensive-summaries-with-vectaras-mmr-reranker/) refers to a technique by which relevant facts provided by the retrieval step are reordered to create a more diverse set of facts. Originally [proposed](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) in 1998, This is critical for example where many matching text chunks, from multiple documents, are in fact very similar or exactly the same. For RAG to work properly we prefer to have a more diverse set of facts.

MMR is often implemented as a reranking step, post retrieval. This can be quite expensive if implemented naively as the algorithm is O(N^2), but at Vectara we took special care to implement this efficiently resulting in super low latency. Vectara’s “diversity\_bias” parameter in the query API controls how the tradeoff between pure relevance (diversity\_bias=0) and full diversity (diversity\_bias=1), and values between 0…1 can provide a good middle ground. We usually recommend values of 0.2 or 0.3 as a good starting point.

### Open Source LLM

The term “open source”, often used to refer to source code that is publicly available under [various licenses](https://opensource.org/licenses/) such as Apache 2.0, MIT, and others has recently been used in the context of [large language models](https://vectara.com/glossary-of-llm-terms/#h-large-language-model-llm) (or LLMs) to refer to such LLMs whose weights are made publicly available for use.

The most notable of these models is [LLAMA3](https://llama.meta.com/llama3/) from meta (which includes the 8B and 70B versions), [Mistral](https://docs.mistral.ai/), Google [Gemma](https://blog.google/technology/developers/gemma-open-models/), Microsoft [Phi3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/), and many [others](https://github.com/eugeneyan/open-llms).

Most open-source LLMs are released under terms that allow commercial use (similar to Apache 2.0 or MIT licenses), but some have additional restrictions. For example, LLAMA3 is released for commercial use but is not allowed to be used by companies with many users such that their MAU (monthly active users) is above 700M.

### Prompt Engineering

The process of crafting, refining, and optimizing the input prompts given to an LLM in order to achieve desired outputs. Prompt engineering plays a crucial role in determining the performance and behavior of models like those based on the GPT architecture.

Given the vast knowledge and diverse potential responses a model can generate, the way a question or instruction is phrased can lead to significantly different results. Some specific techniques in prompt engineering include:

*   **Rephrasing:** Sometimes, rewording a prompt can lead to better results. For instance, instead of asking “What is the capital of France?” one might ask, “Can you name the city that serves as the capital of France?”
*   **Specifying Format:** For tasks where the format of the answer matters, you can specify it in the prompt. For example, “Provide an answer in bullet points” or “Write a three-sentence summary.”
*   **Leading Information:** Including additional context or leading information can help in narrowing down the desired response. E.g., “Considering the economic implications, explain the impact of inflation.”

Prompt engineering is important in the context of [retrieval-augmented-generation](https://vectara.com/glossary-of-llm-terms/#h-retrieval-augmented-generation-rag) as well. To get the best results, a RAG prompt needs to specify a set of instructions to the LLM so that it considers the provided facts in the right way, and can provided the correct response.

### Reranking

In Retrieval-Augmented Generation (RAG), a reranker plays a crucial role in enhancing the relevance of retrieved documents in a two-stage retrieval model. Initially, the retrieval process involves a broad search to gather a large set of potentially relevant documents using a fast and efficient method such as vector embeddings or hybrid search. However, the initial set of documents may contain many irrelevant or marginally relevant entries due to the broad nature of the search. This is where the reranker comes in, acting as the second stage of the retrieval process.

The reranker applies more sophisticated and computationally intensive algorithms, typically based on neural networks, to reassess and reorder the initially retrieved documents. By examining the fine-grained details and contextual relevance of each document, the reranker ensures that the final set of documents presented to the generative model in the final step of the RAG pipeline is of higher quality and most relevant to the query. This two-stage process improves the overall performance and accuracy of RAG systems by combining the efficiency of initial retrieval with the precision of neural reranking, thereby delivering more contextually appropriate and useful information for the generation task.

### Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is an application architecture designed to power LLMs to answer questions about data that is not available in the training set of the LLM.

With Retrieval Augmented Generation, the system allows the LLM to function in the context of data external to its original dataset used for pre-training. The system works as follows:

1.  **Ingest flow:** Data is split into sentences or other chunks of data. Each chunk or sentence is then encoded into a [vector embedding](https://vectara.com/glossary-of-llm-terms/#h-embeddings) (using an Embeddings model like Vectara’s [Boomerang](https://vectara.com/blog/introducing-boomerang-vectaras-new-and-improved-retrieval-model/)), and stored in a vector store.
2.  **Query flow:** When a query is issued, the query is first encoded into its own vector embedding, and those are compared to the data in the vector store. The most relevant sentences or facts from the vector store are retrieved.  Those facts (or sentences) are then provided to a summarizer LLM so that it can respond to the query with this context (set of facts) in mind and provide an accurate response to the query, based on the data.

### Semantic Search and Neural Search

Semantic search is a method used in search algorithms that considers the searcher’s intent and the contextual meaning of terms as they appear in documents. Instead of merely looking at the keywords in a search, a semantic search engine tries to understand the underlying concepts, context, and meaning to deliver more precise results.

For example, when we search for “apple”, a traditional keyword-based search would simply return results containing the word “apple.” It would lack the context to distinguish between results referring to Apple Inc., the technology company, and “apple,” the fruit.

In contrast, a semantic search engine considers the context and the searcher’s intent. For instance, if the user has been searching for smartphone models, a semantic search engine might infer that the user is interested in Apple Inc., and rank those results higher. If the user has been searching for fruit recipes, the engine might instead prioritize results about apples, the fruit.

Semantic search is often based on a deep learning model, and is called “neural search”. With neural search, the text is converted into a form called “vector embeddings” that represent its semantic meaning. Neural search or retrieval engines then compute the similarity between the embeddings of a given query and those of text within a document to generate a ranked list of search results.

### Tokenization

In the context of LLMs, tokenization is a process by which a text string is broken down into smaller units, called tokens. Tokens can be single characters, individual words, or even subwords, and the choice of tokenization strategy depends on the use case and application.

With traditional NLP, tokenization used to be at the word level.

For example the sentence:

“How am I doing today?”

would be translated into 5 words:

\[ “how”, “am”, “I”, “doing”, “today?”\]

One of the issues with this approach is that it’s quite language-specific and can run into issues of large vocabulary size, specification of stop words, and handling of unseen words, among other things.

With LLMs, a new set of tokenization strategies emerged – the most common ones are:

*   [BPE](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) or byte pair encoding. This tokenization method breaks down text into sub-words. Very commonly used in the GPT family of models (GPT-2, GPT-3, etc). See Andrei Karpathy’s [implementation](https://github.com/karpathy/minbpe)
*   [Wordpiece](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt) encoding, developed by Google and used in the original implementation of BERT
*   [Unigram](https://huggingface.co/learn/nlp-course/chapter6/7?fw=pt) encoding, used in Google’s SentencePiece algorithm

This continues to be an area of research, as the type of tokenization used and its implementation can be tied directly to the overall performance of the LLM.

### Transformers

Transformers are a type of neural network architecture that is widely used in natural language processing and language modeling. Transformers apply _[attention](https://arxiv.org/abs/1706.03762)_ which allows the model to consider the importance of all words in a sentence simultaneously. This results in models that are better able to capture the context and long-range dependencies in language.

The transformer architecture consists of an encoder and a decoder. The encoder reads in the input sequence and creates a representation of it. The decoder then uses that representation to generate an output sequence.

Some transformer implementations use an encoder only (e.g. BERT), others use a decoder only (e.g. GPT-3 or Anthropic), while some use both a decoder and encoder (e.g. T5).

Transformers have become very popular for language modeling and have led to huge gains in performance on natural language processing tasks. Models like BERT, GPT-3, and T5 are all based on the transformer architecture. These transformer language models have pushed the state-of-the-art in machine translation, text summarization, question answering, and many other areas. The flexibility, parallel nature, and context handling of transformers have been crucial to their success in natural language processing.

### Transfer Learning

In traditional machine learning, the goal is to train a model to perform a certain task like classification or regression.

With _transfer learning_ you take a pre-trained model and use it (adapt it) to a new task. Because pre-training the model is quite expensive, the idea of transfer learning is the following: can we re-use some of the knowledge learned during the pre-training process and apply it to a new task that benefits from that knowledge?

Transfer earning is not a new idea, it just became more common with modern neural networks that are large and costly to pre-train.

### Vector Store / Vector Database

A vector store is a type of database that stores high-dimensional vector data and provides query capabilities on those vectors such as similarity search, where the goal is to find the most similar documents or items to a given query. The most common types of vector stores are [FAISS](https://ai.meta.com/tools/faiss/#:~:text=FAISS%20(Facebook%20AI%20Similarity%20Search,more%20scalable%20similarity%20search%20functions.), [Annoy](https://github.com/spotify/annoy), and [NMSLib](https://github.com/nmslib/nmslib).

A vector database is a more specialized type of vector store that is optimized for the efficient management of large datasets of vectors. Vector databases typically offer advanced features such as support for multiple vector types, efficient indexing, and high scalability.

There are many stand-alone vector databases available such as Qdrant, Weaviate, Pinecone, Milvus, and Chroma, while existing database systems are slowing introducing vector-database capabilities such as [pv-vector](https://github.com/pgvector/pgvector) for Postgres.

When building GenAI applications from scratch, developers often combine various components like [LangChain](https://github.com/langchain-ai/langchain) or [LlamaIndex](https://github.com/run-llama/llama_index), an embedding provider like OpenAI or Cohere, a summarization or Chat LLM like OpenAI or Bard, and a vector database.

Vectara provides an alternative where many of these components are already integrated into the Vectara platform: the vector database, the embeddings model, and the summarization model. So it’s [not necessary](https://vectara.com/blog/vector-database-do-you-really-need-one/) to use a separate vector store or database.