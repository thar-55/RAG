# RAG

Retrieval Augmented Generation (RAG) has evolved into various types, each designed to address specific challenges and improve the accuracy and efficiency of AI-generated responses. Here are some of the most notable types of RAG:

## Standard RAG

This is the basic form of RAG, where the model retrieves relevant documents from a database in response to a query and generates an output based on the retrieved information[2]. It's suitable for simple applications like FAQ systems or customer support bots where the scope of information is limited to a known set of documents[2].

## Contextual RAG

Contextual RAG enhances the standard approach by adding context to each chunk of information before retrieval[3]. It uses techniques such as contextual embeddings and contextual BM25 to provide chunk-specific explanatory context, improving the accuracy and relevance of the retrieved information[3].

## Speculative RAG

This type employs a hybrid approach combining a specialist RAG drafter and a generalist RAG verifier[3]. The specialist drafts multiple answers based on retrieved documents, while the generalist evaluates and selects the most accurate response[3]. This method is particularly useful for scenarios requiring creative problem-solving or exploration of multiple viewpoints[3].

## Retrieval Augmented Fine-Tuning (RAFT)

RAFT enhances the fine-tuning of language models by integrating a retrieval mechanism that allows the model to access external information during training[3]. This approach improves output quality by expanding the model's knowledge base beyond its initial training data[3].

## Query-based RAG

In this type, the language model generates a query based on the input, which is then used to retrieve relevant information from external knowledge sources[4]. This approach is particularly useful for factual or knowledge-based queries[4].

## Latent Representation-based RAG

This type utilizes latent representations of the input and external knowledge sources to determine the relevance of the retrieved information[4]. It's effective in identifying the most relevant information by comparing latent representations[4].

## Logit-based RAG

This approach uses the raw output values of the language model (logits) to determine the relevance of the retrieved information[4]. The logits are compared with representations of external knowledge sources to select and integrate the most relevant information into the final output[4].

## Modular RAG

Modular RAG introduces enhanced functionalities by integrating a search module for similarity retrieval and adopting a fine-tuning approach in the retriever[5]. It allows for either a serialized pipeline or end-to-end training, addressing specific challenges more effectively[5].

## Hybrid Search RAG

This type optimizes RAG performance by integrating various search techniques, including keyword-based, semantic, and vector searches[5]. It leverages each method's strengths to ensure consistent retrieval of relevant and context-rich information[5].

## Recursive Retrieval RAG

Recursive Retrieval involves acquiring smaller chunks initially to capture key semantic meanings, then providing larger chunks with more contextual information to the language model later[5]. This two-step retrieval method balances efficiency and contextually rich responses[5].

These various types of RAG demonstrate the ongoing evolution and refinement of the technology, each addressing specific needs and use cases in the field of AI-powered information retrieval and generation[1][2][3][4][5].

Citations:
[1] https://www.weka.io/learn/guide/ai-ml/retrieval-augmented-generation/
[2] https://humanloop.com/blog/rag-architectures
[3] https://research.aimultiple.com/retrieval-augmented-generation/
[4] https://www.cloudraft.io/what-is/retrieval-augmented-generation
[5] https://blog.jayanthk.in/types-of-rag-an-overview-0e2b3ed71b82?gi=4654eb587df4
[6] https://aws.amazon.com/what-is/retrieval-augmented-generation/?trk=faq_card
[7] https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts
[8] https://redis.io/glossary/retrieval-augmented-generation/
[9] https://www.smashingmagazine.com/2024/01/guide-retrieval-augmented-generation-language-models/
[10] https://www.linkedin.com/pulse/exploring-12-types-retrieval-augmented-generation-rag-gaurang-desai-3lmde
[11] https://www.marktechpost.com/2025/01/10/top-9-different-types-of-retrieval-augmented-generation-rags/
[12] https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/


Retrieval-Augmented Generation (RAG) has indeed evolved into various architectures to address specific challenges and optimize performance. Here's an overview of some notable RAG types:

## Naive RAG

This is the foundational RAG approach, involving a straightforward process:

- **Indexing**: Data is extracted, cleaned, and divided into manageable chunks, which are then transformed into vectors using embedding models.
- **Retrieval**: Upon receiving a user query, the system encodes it into a vector and performs a similarity search to find the most relevant data chunks.
- **Generation**: The retrieved chunks are combined with the user query to form a prompt, which is fed to the LLM to generate a response[1].

While simple, Naive RAG can suffer from limitations such as low precision in retrieval and potential inaccuracies if relevant chunks are not properly retrieved.

## Advanced RAG

Advanced RAG incorporates additional processes to overcome the shortcomings of Naive RAG:

- **Pre-Retrieval Processes**: Enhances data quality by removing irrelevant information, resolving ambiguities, and updating outdated documents. Query rewriting techniques are also employed to improve generation quality.
- **Post-Retrieval Processes**: Utilizes techniques like re-ranking retrieved chunks based on contextual similarity and prompt compression to improve response quality[1].

These enhancements lead to more accurate and contextually relevant outputs.

## Modular RAG

Modular RAG introduces enhanced functionalities by integrating additional modules and adopting a fine-tuning approach:

- **Hybrid Search**: Combines various search techniques, including keyword-based, semantic, and vector searches, to ensure consistent retrieval of relevant information.
- **Recursive Retrieval**: Acquires smaller chunks initially to capture key semantic meanings, followed by larger chunks with more contextual information[5].

This architecture provides greater adaptability and addresses specific challenges more effectively.

## Speculative RAG

Speculative RAG introduces a novel approach that decomposes RAG tasks into two separate steps:

1. **Drafting**: A small specialized RAG drafter generates multiple drafts from diverse document subsets.
2. **Verification**: A large generalist LM verifies and refines the drafts.

This method has shown substantial improvements in both quality and speed of final output generation, with accuracy gains up to 12.97% while reducing latency by 51% compared to standard RAG systems[23].

## Conversational RAG

Designed to facilitate natural, interactive dialogue, Conversational RAG creates contextually relevant responses in real-time:

- **Conversation Context Analysis**: Analyzes current and past interactions to maintain context.
- **Dynamic Response Generation**: Generates engaging responses based on the conversation flow[1].

This architecture is ideal for customer support chatbots and virtual assistants where conversational engagement is essential.

Each of these RAG architectures offers unique advantages tailored to specific applications, contributing to the ongoing evolution and enhancement of AI-driven information retrieval and generation systems.

Citations:
[1] https://www.weka.io/learn/guide/ai-ml/retrieval-augmented-generation/
[2] https://www.signitysolutions.com/blog/rag-architecture-techniques-use-cases
[3] http://arxiv.org/pdf/2410.12837.pdf
[4] https://www.promptingguide.ai/research/rag
[5] https://blog.jayanthk.in/types-of-rag-an-overview-0e2b3ed71b82?gi=4654eb587df4
[6] https://www.leewayhertz.com/advanced-rag/
[7] https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview?WT.mc_id=DT-MVP-10953
[8] https://winder.ai/llm-architecture-rag-implementation-design-patterns/
[9] https://www.smashingmagazine.com/2024/01/guide-retrieval-augmented-generation-language-models/
[10] https://www.nvidia.com/en-in/glossary/retrieval-augmented-generation/
[11] https://www.marktechpost.com/2025/01/10/top-9-different-types-of-retrieval-augmented-generation-rags/
[12] https://www.linkedin.com/pulse/rag-architecture-deep-dive-frank-denneman-4lple
[13] https://www.databricks.com/glossary/retrieval-augmented-generation-rag
[14] https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/
[15] https://arxiv.org/html/2410.15944v1
[16] https://aws.amazon.com/what-is/retrieval-augmented-generation/?trk=faq_card
[17] https://www.k2view.com/what-is-retrieval-augmented-generation
[18] https://www.researchgate.net/publication/385009890_A_Comprehensive_Survey_of_Retrieval-Augmented_Generation_RAG_Evolution_Current_Landscape_and_Future_Directions
[19] https://www.glean.com/blog/rag-revolutionizing-ai-2024
[20] https://humanloop.com/blog/rag-architectures
[21] https://attri.ai/blog/retrieval-augmented-generation-rag-architecture
[22] https://www.k2view.com/blog/rag-architecture/
[23] https://research.google/blog/speculative-rag-enhancing-retrieval-augmented-generation-through-drafting/

Here are additional types of Retrieval Augmented Generation (RAG), expanding on the list you provided:

## **Logit-based RAG**  
Logit-based RAG optimizes retrieval by adjusting token probabilities based on retrieved documents. Instead of relying solely on textual matches, it fine-tunes the logits (output probabilities) of the model to enhance alignment with retrieved knowledge. This improves response accuracy, especially for ambiguous or low-confidence queries.

## **Multi-Hop RAG**  
Multi-Hop RAG retrieves information in multiple steps, where each retrieval phase refines the search based on previous results. This is useful for answering complex, multi-part queries that require combining information from different sources.

## **Memory-Augmented RAG**  
This approach incorporates a persistent memory mechanism that stores frequently retrieved knowledge. Instead of always querying a database, the model first checks its memory, reducing redundant retrieval and improving efficiency over time.

## **Agentic RAG**  
Agentic RAG integrates AI agents capable of iterative querying, decision-making, and reasoning. The model autonomously refines its retrieval process based on previous attempts, making it ideal for dynamic and evolving knowledge environments.

## **Neural RAG**  
Neural RAG replaces traditional retrieval mechanisms (like BM25 or vector search) with fully neural-based retrieval models. It leverages deep learning techniques such as contrastive learning to improve the relevance of retrieved information.

## **Cross-Encoder RAG**  
This method enhances retrieval by scoring document-query pairs using a cross-encoder model instead of a standard retriever. By jointly encoding the query and documents, it significantly improves precision in selecting relevant content.

Thank you for providing these additional types of Retrieval Augmented Generation (RAG). I'll expand on each of these types and provide a comparison table as requested.

## Logit-based RAG

Logit-based RAG enhances the retrieval process by adjusting token probabilities based on retrieved documents. This approach fine-tunes the logits (output probabilities) of the model to better align with retrieved knowledge, improving response accuracy for ambiguous or low-confidence queries[3].

## Multi-Hop RAG

Multi-Hop RAG implements a step-wise retrieval process, where each phase refines the search based on previous results. This method is particularly effective for complex, multi-part queries that require synthesizing information from various sources[3].

## Memory-Augmented RAG

Memory-Augmented RAG incorporates a persistent memory mechanism to store frequently retrieved knowledge. By first checking its memory before querying a database, this approach reduces redundant retrieval and improves efficiency over time[3].

## Agentic RAG

Agentic RAG integrates AI agents capable of iterative querying, decision-making, and reasoning. These agents can autonomously refine the retrieval process based on previous attempts, making it well-suited for dynamic and evolving knowledge environments[4].

## Neural RAG

Neural RAG replaces traditional retrieval mechanisms with fully neural-based retrieval models. By leveraging deep learning techniques such as contrastive learning, it aims to improve the relevance of retrieved information[3].

## Cross-Encoder RAG

Cross-Encoder RAG enhances retrieval by scoring document-query pairs using a cross-encoder model instead of a standard retriever. This joint encoding of queries and documents significantly improves precision in selecting relevant content[3].

Here's a comparison table for these RAG types:

| RAG Type | Key Feature | Best Use Case | Main Advantage |
|----------|-------------|----------------|----------------|
| Logit-based RAG | Adjusts token probabilities | Ambiguous or low-confidence queries | Improved response accuracy |
| Multi-Hop RAG | Step-wise retrieval | Complex, multi-part queries | Better information synthesis |
| Memory-Augmented RAG | Persistent memory mechanism | Repetitive queries or tasks | Reduced redundant retrieval |
| Agentic RAG | AI agents for iterative querying | Dynamic knowledge environments | Autonomous refinement |
| Neural RAG | Fully neural-based retrieval | Large-scale, diverse datasets | Improved relevance of retrieved info |
| Cross-Encoder RAG | Joint query-document encoding | Precision-critical applications | Higher precision in content selection |

These advanced RAG types demonstrate the ongoing evolution in the field of AI-powered information retrieval and generation, each addressing specific challenges and use cases in different scenarios.

Citations:
[1] https://www.weka.io/learn/guide/ai-ml/retrieval-augmented-generation/
[2] https://www.plainconcepts.com/rag-vs-kag/
[3] https://research.aimultiple.com/retrieval-augmented-generation/
[4] https://www.akira.ai/blog/rag-vs.-agentic-rag
[5] https://blog.jayanthk.in/types-of-rag-an-overview-0e2b3ed71b82?gi=4654eb587df4
[6] https://dev.to/codemaker2015/the-ultimate-guide-to-retrieval-augmented-generation-rag-5e6e
[7] https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/
[8] https://learn.microsoft.com/lb-lu/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide
[9] https://cloud.google.com/use-cases/retrieval-augmented-generation?e=48754805&hl=en
[10] https://community.aws/content/2fpZlwwebByVkQH7uH5yD8fFFsQ/choose-the-right-rag-architecture-for-your-aws-genai-app?lang=en
[11] https://www.linkedin.com/pulse/exploring-12-types-retrieval-augmented-generation-rag-gaurang-desai-3lmde
[12] https://www.chatbees.ai/blog/rag-architecture-llm
[13] https://www.databricks.com/glossary/retrieval-augmented-generation-rag
[14] https://www.k2view.com/what-is-retrieval-augmented-generation
[15] https://blog.lewagon.com/skills/exploring-rag-architecture-how-retrieval-augmented-generation-is-revolutionizing-genai/
[16] https://repositum.tuwien.at/bitstream/20.500.12708/202324/1/Oroz%20Tin%20-%202024%20-%20Comparative%20Analysis%20of%20Retrieval%20Augmented%20Generator%20and...pdf
[17] https://www.researchgate.net/publication/385167532_Comparative_Analysis_of_Retrieval-Augmented_Generation_RAG_based_Large_Language_Models_LLM_for_Medical_Chatbot_Applications
[18] https://aws.amazon.com/what-is/retrieval-augmented-generation/?trk=faq_card
[19] https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview?WT.mc_id=DT-MVP-10953

