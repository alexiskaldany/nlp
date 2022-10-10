#%%
# ------------------Import Library----------------------------
import pandas as pd
import spacy
# Download and load the NER.xlsx a from BB read it through python.
# 2. Find all the location, names, cities and generally all entities and replace
# them with asterisk.
# 3. Save all the redacted version of the sentence into a column and save it into
# a new excel file.
#%%
# ------------------Main Loop----------------------------
df = pd.read_excel('/Users/alexiskaldany/school/nlp/code/scatch/NER.xlsx')
nlp = spacy.load('en_core_web_sm')


# ------------------Part i----------------------------

def replace_ner(mytxt,nlp):
    clean_text = mytxt
    doc = nlp(mytxt)
    for ent in reversed(doc.ents):
        clean_text = clean_text[:ent.start_char] + "*"* len(ent.text) + clean_text[ent.end_char:]
    return clean_text

def replace_replace_ner(mytxt,nlp):
    clean_text = mytxt
    doc = nlp(mytxt)
    for ent in doc.ents:
        clean_text = clean_text.replace(ent.text,"*" * len(ent.text))
    return clean_text

df["redacted_0"] = df["Sentences"].apply(lambda x: replace_ner(x,nlp)) 
df["redacted_1"] = df["Sentences"].apply(lambda x: replace_replace_ner(x,nlp)) 
df.to_excel('/Users/alexiskaldany/school/nlp/code/scatch/NER_redacted.xlsx',index=False)

print(df.head())
# ------------------Part ii----------------------------

# ------------------Part iii----------------------------

# %%
