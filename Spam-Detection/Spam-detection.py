from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

CORPUS = [
  'Dear Sir, you have won a million. Come collect your gift',
  'Do you want to try these pills that make you look better? it only costs 100$',
  'Hi, you have been selected for an interview!',
  'Sale! Clearance!',
  'Your account has been hacked!',
  'Hello Sir, I want to inform you that you have won the OLG jackpot',
  'Please enter your username and password',
  'Hi Alice, how are you doing today?',
  'Hi Alice!',
  'Congratulations!'
  'How are you?',
  ''
]

# Spam = 0 
# Non spam = 1
y = [0,0,1,1,0,0,0,1,1,1,1]

vectorizer = CountVectorizer()
vectorizer.fit(CORPUS)

x = vectorizer.transform(CORPUS)

classifier = MultinomialNB()
classifier.fit(x,y)

tests = [
    "Dear sir, you have won one million money",
    "Your account has been hacked"
]
test_vectors = vectorizer.transform(tests)
print(classifier.predict(test_vectors))
