import spacy
from spacy import displacy
with open("data/genesis_1.txt", "r") as f:
    text = f.read()

nlp = spacy.load("en_core_web_trf")

doc = nlp(text)
sentences = list(doc.sents)
print(sentences[0])
for token in list(doc.sents)[0]:
    print(
        token.text,
        token.lemma_,
        token.pos_,
        token.tag_,
        token.dep_,
        token.shape_,
        token.is_alpha,
        token.is_stop,
    )
displacy.serve(list(doc.sents)[0], style="dep",port=5001,options={"compact":True})