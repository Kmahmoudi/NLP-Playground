
import gensim

# Assuming you have loaded your trained Word2Vec model
model = gensim.models.Word2Vec.load('./model.wv')

# Generate text
while True:
    print("word:")
    query=input()
    generated_text = model.wv.most_similar(positive=[query], topn=10)
    print("similar words:")
    for word, similarity in generated_text:
        print(word,end=' ')
    print("\n")