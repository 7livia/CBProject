import pandas as pd
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the CSV file into a DataFrame
df = pd.read_csv('C:/Users/Livia/Desktop/dataset.csv')

# Convert DataFrame rows into a list of dictionaries (input-output pairs)
dataset = df.to_dict(orient='records')

number_variations = {
    "1": ["një", "1"],
    "2": ["dy", "2"],
    "3": ["tre", "3"],
    "4": ["katër", "4"],
    "5": ["pesë", "5"],
    "6": ["gjashtë", "6"],
    "7": ["shtatë", "7"],
    "8": ["tetë", "8"],
    "9": ["nëntë", "9"],
    "10": ["dhjetë", "10"]
}

# Dictionary of possible synonyms for augmentation
synonyms = {
    "fëmija": ["vajza", "djali"],
    "çfarë": ["ça", "cila"],
    "ndihmoj": ["asistoj", "mbështes"],
    "porosinë": ["porosinë tuaj", "blerjen tuaj"],
    "kontrolloni": ["verifikoni", "kontrolloni", "shikoni"],
    "statusin": ["gjendjen", "situatën"],
    "dorëzimin": ["dërgesën", "pranimin"],
    "proçes": ["procedurë"],
    "angli" :["greqi" ,"itali" , "spanjë" , "milano" , "romë" , "londër" , "athinë" ],
}

# Typographical errors to simulate user input variations
typo_variations = {
    "porosia": ["porosia", "prorosia"],
    "statusin": ["statussin", "statuin"],
    "mund": ["mund", "muund"],
    "fëmija": ["fmija", "fmia", "femija"],
    "çfarë": ["ca", "cila", "cfarë","cfare"],
    "shqiptarë" : ["shiptar","shqiptare"],
    "aplikuar" : ["apliku"],
    "skaduar" :["skadu"],
    "përshpejtuar": ["pershpejtu" , "pershpejtume" , "pershpejtuar"],
    "Letërnjoftimi" : ["leternjoftim" , "leternjoftimi"],
    "zyrë" : ["zyr" , "zyra"],    
    "online" : ["onlajn"],
    "kartë" : ["kart"],
    "vjeç" : ["vjec" , "vjeq" ],
    "ë" : ["e"],
    "ç" : ["c"],
    "të jem":["te jem" , "tjem" , "me qen" , "me qan"],
    "pasaportë" : ["pashaport" , "pashaportë"],
    "rinovim" : ["rifreskim" , "riaplikim"],
    "lëshuar" :["leshuar" , "leshu" , "lshu"],
    "humbur" :["humb"],
    "vërtetimin" :["vertetimin" , "vertetim"],
    "tërheq" :["terheq" , "merr" , "terhek"],
    "prokurë" :["prokur"],
    "pjesëtari" :["pjestar" , "antar" , "anetar"],
}

# Function to augment sentence with synonym replacement
def synonym_replacement(sentence, synonym_dict):
    words = sentence.split()
    new_sentence = []
    
    for word in words:
        if word in synonym_dict:
            new_word = random.choice(synonym_dict[word])
            new_sentence.append(new_word)
        else:
            new_sentence.append(word)
    
    return " ".join(new_sentence)

# Function to introduce minor typographical variations
def typo_augmentation(sentence, typo_dict):
    words = list(sentence)  # Treat the sentence as a list of characters
    new_sentence = []
    
    for char in words:
        if char in typo_dict:
            new_char = random.choice(typo_dict[char])
            new_sentence.append(new_char)
        else:
            new_sentence.append(char)
    
    return "".join(new_sentence)


# Function to rephrase sentence (manual input for now)
def rephrase_sentence(sentence):
    rephrased = {
        "Femija im do të aplikojë për letërnjoftim elektronik për herë të parë. Cfare dokumentesh duhen?": "Cfare dokumentesh duhen per aplikimin e leternjoftimit elektronik?",
        "Sa është vlefshmëria e leternjoftimit elektronik nga dita që tërheq dokumentin?": "Sa vlen leternjoftimi elektronik nga dita qe terheq dokumentin?",
    }
    return rephrased.get(sentence, sentence)

# Main function to augment dataset
def augment_dataset(dataset):
    augmented_dataset = []
    
    for pair in dataset:
        input_sentence = pair['input']
        output_sentence = pair['output']
        
        # Apply synonym replacement for input and output
        augmented_input_syn = synonym_replacement(input_sentence, synonyms)
        augmented_output_syn = synonym_replacement(output_sentence, synonyms)
        
        # Apply typographical variation for input
        augmented_input_typo = typo_augmentation(input_sentence, typo_variations)
        
        # Apply manual rephrasing for input
        augmented_input_rephrase = rephrase_sentence(input_sentence)
        
        # Create augmented data points
        augmented_dataset.append({"input": augmented_input_syn, "output": augmented_output_syn})
        augmented_dataset.append({"input": augmented_input_typo, "output": output_sentence})
        augmented_dataset.append({"input": augmented_input_rephrase, "output": output_sentence})
    
    return augmented_dataset

# Function to detect numbers in the sentence and replace them
def number_augmentation(sentence, number_dict):
    # Use regex to find all numbers in the sentence
    def replace_number(match):
        num = match.group(0)
        # If the number is in our variation dictionary, replace it
        return random.choice(number_dict.get(num, [num]))

    # Replace all digits in the sentence using the number dictionary
    return re.sub(r'\b\d+\b', replace_number, sentence)

# List of numbers you want to replace with
number_replacements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Function to replace numbers in a sentence with random numbers from the list
def replace_numbers(sentence, number_list):
    # Use regex to find numbers in the sentence
    def replace_number(match):
        # Choose a random number to replace the existing number
        return str(random.choice(number_list))
    
    # Replace all numbers in the sentence
    return re.sub(r'\b\d+\b', replace_number, sentence)

# Function to augment dataset by creating multiple variations with different numbers
def augment_with_number_variations(dataset, number_list):
    augmented_dataset = []
    
    for pair in dataset:
        input_sentence = pair['input']
        output_sentence = pair['output']
        
        # Generate 3 variations for each input-output pair by changing numbers
        for _ in range(3):  # You can adjust the number of variations here
            augmented_input = replace_numbers(input_sentence, number_list)
            augmented_output = replace_numbers(output_sentence, number_list)
            augmented_dataset.append({"input": augmented_input, "output": augmented_output})
    
    return augmented_dataset

# Augment the loaded dataset from the CSV
augmented_data = augment_dataset(dataset)

def normalize_text(text):
    """Function to normalize text (lowercase, remove punctuation, handle ë/e)."""
    # Replace ë with e for convenience
    text = text.replace("ë", "e")
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', '', text).strip()
    return text

# Convert DataFrame rows into a list of dictionaries (input-output pairs)
dataset = df.to_dict(orient='records')

# Function to preprocess text (convert to lowercase, remove special characters)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Preprocess the dataset inputs
for pair in dataset:
    pair['input_preprocessed'] = preprocess_text(pair['input'])

# Function to extract keywords from the user's input and match with dataset
def keyword_matching(user_input, dataset):
    user_input_preprocessed = preprocess_text(user_input)
    
    # Split user's input into words
    user_input_words = set(user_input_preprocessed.split())

    # Look for matching keywords in the dataset
    for pair in dataset:
        dataset_input_words = set(pair['input_preprocessed'].split())
        
        # Check if there are any common words between user input and dataset input
        if user_input_words & dataset_input_words:
            return pair['output']

# Function to find the best match for the user's input
def find_response(user_input, dataset):
    normalized_input = normalize_text(user_input)
    
    # Search for a match in the dataset
    for pair in dataset:
        normalized_dataset_input = normalize_text(pair["input"])
        if normalized_input == normalized_dataset_input:
            return pair["output"]
    
    # If no exact match is found, return a fallback response
    return "Më falni, pyetja juaj nuk eshte e qarte."


# Prepare the dataset inputs for similarity comparison
inputs = [pair['input_preprocessed'] for pair in dataset]

# Function to find the most similar dataset entry using TF-IDF and cosine similarity
def find_most_similar_response(user_input, dataset):
    user_input_preprocessed = preprocess_text(user_input)
    
    # Combine user input with dataset inputs
    all_inputs = inputs + [user_input_preprocessed]
    
    # Use TF-IDF vectorizer to convert text into numerical vectors
    vectorizer = TfidfVectorizer().fit_transform(all_inputs)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity between user input and dataset inputs
    similarity_scores = cosine_similarity([vectors[-1]], vectors[:-1])
    
    # Get the index of the most similar input
    most_similar_index = similarity_scores.argmax()
    
    return dataset[most_similar_index]['output']

# Terminal interaction loop with TF-IDF similarity
def chatbot_loop_with_similarity(dataset):
    print("Mirë se vini! Si mund tju ndihmoj?")
    
    while True:
        user_input = input("Ju: ")
        
        if user_input.lower() in ["dal", "exit", "quit"]:
            print("Faleminderit për përdorimin e shërbimit!")
            break
        
        response = find_most_similar_response(user_input, dataset)
        print(f"Chatbot: {response}")

# Run the chatbot loop
chatbot_loop_with_similarity(dataset)