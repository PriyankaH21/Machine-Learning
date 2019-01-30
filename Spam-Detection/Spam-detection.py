from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

CORPUS = [
  'Dear Sir, you have won money $$$$. Come collect your gift',
  'Do you want to try these pills that make you look better? it only costs 100$',
  'Hi, you have been selected for an interview!',
  'Sale! Clearance!'
]
