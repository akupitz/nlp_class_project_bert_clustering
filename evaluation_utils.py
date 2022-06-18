# todo: write evaluation utils:
"""
1. check number of output topics
2. get topic sizes
3. get_info about a topic: topic_words, word_scores, topic_document_indices = model.get_topics(77) <from top2vec github page>
4. search topics with a keyword: topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["medicine"], num_topics=5)
5. generate word clouds:
topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["medicine"], num_topics=5)
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)
6. search documents by topic: documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=48, num_docs=5)
"""

