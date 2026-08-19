from src.reranker import RerankingRetriever
from src.generator import generate

class Pipeline:

    def __init__(self, k = 10, top_k = 5):
        self.retriver = RerankingRetriever(fetch_k = k, top_k = top_k)

    def invoking(self, query):
        context = self.retriver.invoke(query)

        good_context = [doc.page_content for doc in context]

        answer = generate(query, good_context)

        return {
            'query':query,
            'context':good_context,
            'answer':answer
        }

if __name__== '__main__':
    rag = Pipeline()
    result = rag.invoking('What is Online Eval')
    print(result)
    
    