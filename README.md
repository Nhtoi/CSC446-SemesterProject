
    Kevin Ruiz, Jaskaran Chahal
    I. PROJECT DEFINITION
    The goal of this project is to develop and train an AI
    agent for summarizing forum conversations. The first step
    involves extracting, cleaning, and processing discussion data
    to generate meaningful summaries while maintaining contextual integrity. Automated summarization can help streamline
    information retrieval and enhance comprehension of online
    discussions.
    A. Problem Statement
      Online discussion forums generate vast amounts of usergenerated content, making it difficult to extract key insights
    efficiently. This issue is especially relevant in technical or
    research-related forums where information is scattered across
    lengthy threads. Our objective is to leverage AI-based summarization techniques to automatically distill the most relevant
    information from discussions, improving usability and knowledge extraction.
    B. Significance
      Summarizing discussions enhances information retrieval,
    aids academic and industry research, and improves user
    engagement by providing concise insights. This is particularly useful for professionals, researchers, and casual users
    who need quick access to key points without reading entire
    discussions. Automating this process can also contribute to
    advancements in natural language processing (NLP) and AIdriven content summarization.
    C. Challenges
      • Extracting relevant data from websites using APIs (Reddit
    API, PRAW) while adhering to data privacy and ethical
    guidelines.
      • Cleaning and preprocessing textual data (removing URLs,
    special characters, and normalizing text) to enhance
    model performance.
      • Implementing and training an AI model (Transformers/LSTMs) capable of capturing contextual nuances in
    online discussions.
      • Evaluating different summarization techniques (extractive
    vs. abstractive) to determine the most effective approach.
      • Optionally, developing a front-end interface for improved
    accessibility and usability of generated summaries
    II. RELATED WORK
    Research in text summarization has explored extractive
    and abstractive techniques. Extractive approaches focus on
    selecting key sentences directly from the text, while abstractive
    methods generate new sentences that convey the main ideas.
    Methods such as TextRank, LexRank, and transformer-based
    models (e.g., BERTSUM) have demonstrated effectiveness in
    summarizing textual content.
      A. Existing Approaches
      Traditional summarization techniques rely on statistical and
      graph-based ranking methods, such as TextRank, which scores
      sentences based on their importance within a document. More
      advanced techniques leverage deep learning and NLP to improve the coherence and informativeness of summaries.
      B. Cutting-edge Methods
      Recent advancements include transformer-based architectures like BERTSUM and PEGASUS, which leverage pretrained language models to generate context-aware summaries.
      Weakly supervised learning approaches, such as heuristicbased sentence selection, help reduce manual annotation efforts. This aligns with our project’s goal, as we aim to
      minimize the need for manually labeled training data while
      still achieving high-quality summarization results.
    III. DATASET
      Since no annotated corpus is readily available, we will
      scrape discussion data from Reddit using the Reddit API
      (PRAW). This dataset will consist of various subreddit discussions, allowing us to develop and test our summarization models on real-world user interactions. We will focus on subreddits
      that feature long-form discussions, such as r/MachineLearning,
      r/AskScience, and r/Technology.
        A. Data Collection
          • Utilize the Reddit API (PRAW) to collect posts and
            comments from selected subreddits.
          • Store collected data in a structured format for preprocessing and analysis.
          • Ensure compliance with Reddit’s data usage policies and
          ethical considerations.
          B. Data Cleaning and Processing
          • Remove redundant, non-textual elements (e.g., URLs,
          emojis, special characters) to enhance model input quality.
          •  Normalize text by converting it to lowercase and eliminating stopwords that do not contribute to meaning.
          • Apply tokenization and sentence segmentation for structured processing.
          • Explore Named Entity Recognition (NER) to retain critical entities while summarizing text.
      IV. EVALUATION PLAN
      To assess our summarization approach, we will implement
      the following evaluation strategies:
          • ROUGE Scores: Measure summary quality by comparing generated summaries with human-written summaries,
          evaluating recall, precision, and F1-score.
          • Ablation Studies: Compare different model variations
          (e.g., fine-tuned vs. pre-trained models) to understand
          their effectiveness.
          • Efficiency Metrics: Measure inference time and computational efficiency for practical deployment considerations.
        V. REFERENCES
      X. Liu and Y. Lapata, ”Query-based Summarization of Discussion
      Threads,” Natural Language Engineering, 2020.
     A. Radev, ”Multi-stage Online Forum Summarization,” ResearchGate,
    2020. https://www.researchgate.net/publication/341298724 Multistage
    Online Forum Summarization#pff
