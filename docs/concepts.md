# Lessons & Conceptual Understanding
As much as building a product is important, the understanding behind the working product is just as important, if not even more so.

The following sections will contain a snapshot of documented questions and lessons that I've encountered along my learning journey, and my attempts at understanding and learning from it while applying these concepts in practice.

## Retrieval-Augmented Generation (RAG)
Using RAG is like having an open-book exam, where the model is able to reference material that it may not have been trained on, to answer a question based on this material.

It works by:
1. Receiving a question in natural language format
2. It converts the question into vectors (or otherwise known as embeddings)
3. Using the vectors, it looks up a vector database to look for similar chunks (of information)
4. It will retrieve the top X most relevant chunks
5. These chunks will be combined, along with the original question

## Why are questions converted into a vector?
Documents may contain words that don't form an exact match to the question that was posed, for example: A document could say "To process payments, use the Payment Intent API", while the question was: "How do I accept payments for a credit card?".

The traditional keyword search would fail since those words don't appear in the document.

Vectors provide the ability to convert text into numbers which represent meaning, rather than just a match on a word. Similar meanings would have similar numbers, and this allows looking up similar words or meanings together, without having a close match on a word.

Meanings can be calculated based on measures such as cosine similarity ```(dot_product(vector1, vector2) / (length(vector1) * length(vector2)))```, and this provides a score between -1 and 1. A score of 1 indicates identical meaning, 0 indicates no relatedness, and -1 indicates opposite meaning. There are other measures that can be used as well.

Note that vectors are only used for **searching**. Once the relevant chunks are found, everything gets converted back into text.

## How many chunks do I need to return?
We can use less chunks when the questions are specific, and /or when documents are well-structured, and concise. Less chunks can also be used if cost is a concern (though follow-up questions may be used which could also add to using additional cost).

More chunks are required when questions tend to be broader, and need to combine information from multiple sections. More chunks can also be used when accuracy is more important than the cost itself.

A simple recommendation could be:
- 1 to 2 chunks for very specific questions
- 3 to 5 chunks for a balance, which is good for most cases
- 10+ chunks for when it is a complex, multi-part question

In Claude's case, 5 is a good default since most questions require 2 to 3 chunks to answer fully. Retrieving 5 chunks provides some buffer for the model to use, since not all results are perfect. 5 chunks also fits comfortably in Claude's context window, and is a good balance between accuracy and cost.

## What is a context window?
This is the maximum amount of text (measured in tokens, not words) that an LLM (large language model) can "see" and process at once. It's like the "attention span" or "working memory" of a brain.

## How does Claude process the natural language prompt?
The model first breaks the prompt into tokens through a process called tokenization. Internally these tokens are then represented as vectors. Note that these vectors are **NOT** for searching, but instead, used for internal embeddings. These are learned representations which are specific to Claude's neural network.

At this point, Claude will "pay attention" to the relevant parts of the context and question provided, and tries to establish a connection between them. It then builds its own understanding of the relationship between them, and generates a response token by token.

## How does Claude understand the context and question?
Given the context and question in tokenized form, Claude will review all tokens simultaneously, connecting the words from the question and finding its own understanding to the words in the context.

When it comes to generating the actual response, it will create probabilities (based on patterns it has learned during model training, the context that was provided to it, grammatical patterns, and some weighting by temperature) to think of what the best token is to answer the user. After it picks its first token, it then considers this first token as part of its next consideration for the next token, and so on and so forth.

So at each step, Claude will look at:
1. The original context (the documentation provided to it)
2. The question posed to it
3. Everything Claude has generated so far
4. Patterns it learned from its training data

## What is temperature?
Temperature is how random or creative the model's outputs get to be. It is a number typically between 0 and 2 that affects how Claude chooses the next token.

The following is a general guide of temperature values and how it causes the model to behave:

- 0 is no randomness, it always picks the token with the highest probability
- 0.2 to 0.4 is factual and consistent, good for RAG.
- 0.7 to 0.8 is medium, and is the default for ChatGPT. It has varied phrasing but with the same core information. This is good for chat.
- 1 is balanced, it picks the token based on its probability. e.g. a token that is 15% probability to be correct will be chosen 15% of the time
- 1.5 is more creative, creating very different answers but potentially at risk of hallucinating
- 2 is very random, experimental and chaotic

## What is chunking?
Chunking is breaking large documents down into smaller sections which can be embedded as vectors for searching relevant contexts. Chunking is important as it allows large documents (relevant parts of it) to be fitted in an embedding model and lets it be specific enough to be used for good retrieval.

There are better ways of chunking documents, such as:
- Splitting at paragraph boundaries
- Adding overlap between chunks for context
- Preserving semantic meanings

This allows chunks to be complete thoughts, and not just parts of text that are isolated mid-sentence. The overlap maintains context and allows for better retrieval accuracy.

## Chunking for HTML markup content
One of the lessons I learnt when trying to chunk my documents, was noticing that I was exceeding the token count during my chunking and vectorization process. A quick look showed me how simple semantic chunking didn't help when processing documentation that contained markup content such as HTML and CSS. To make things more complicated, there were sections which sometimes contained code fragments that were meant to be kept and returned to the user, which were somewhat like quotes that needed to be maintained _ad verbatim_.

At first I attempted to use BeautifulSoup to strip away the tags simply, but it didn't work quite so well. Sometimes partially opened tags remained in the context, while some other times I had whitespaces which weren't trimmed. Clearly using the parser as is didn't help, plus this caused me to lose the code fragments that were meant to help users understand how to use this.

After some further investigation with Claude Code, it turns out one of the better ways to deal with this is to clean the data in a multi-stage process. The first stage starts by cleaning most of the structural bloat from the content that adds little value to the retrieval augmentation, **during** the data collection process (so things like `<script>`, `<style>`, `<nav>`, `<footer>`, HTML comments). While doing this, we should also take note to keep things that make semantic sense, such as paragraphs (demarcated as `<p>`), list structures (indicated with `<li>`), and code fragments (`<code>`) intact. Then we normalize the remaining content by removing additional whitespaces, newlines, and converting it all to plaintext.

The second stage works by processing the content while chunking takes place, and applying our own normalization techniques at the processing stage without losing the original content that we had. You can read more about it in the [text cleaning strategy](./text_cleaning_strategy.md).
