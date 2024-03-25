# every file in this list must have an associated text file
names = [
  "aita", 
  "cs",
  "attention",
  "gender_studies", 
  "holocaust_textbook",
  "1984",
  "bible",
]

# relevant to ./data_processing
filepaths = ['../data/'+name+'.txt' for name in names]