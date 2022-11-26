# nlp



```

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip3 freeze > requirements.txt
```

`python -m venv .venv`
`source .venv/bin/activate`
`pip install --upgrade pip`
`pip install -r requirements.txt`

## Final Project Ideas

Diagnostic Questions

Paragraph of text with 4 potential outputs. classification?
Need to get large dataset (scrape potentially?)

AWS
for EC2: 
`Alexis-Labadie Kaldany`
`akaldany@gwu.edu`

ssh -i /Users/alexiskaldany/Personal/nlp_key.pem ec2-3-95-222-153.compute-1.amazonaws.com

----------------
## Midterm

Write Regex guide
Create code for SVD/Last lecture stuff
Write guide for nltk
Write guide for spaCy

ssh -i /Users/alexiskaldany/Personal/nlp_key.pem root@ec2-34-207-123-132.compute-1.amazonaws.com

scp -i /Users/alexiskaldany/Personal/nlp_key.pem /Users/alexiskaldany/school/nlp22_final/covid_articles_raw.csv ubuntu@ec2-3-95-222-153.compute-1.amazonaws.com:/home/ubuntu/nlp22_final
