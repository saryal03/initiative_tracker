from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_login import LoginManager, login_required, current_user


# Specify paths to your saved model and tokenizer
model_path = '/initiativetracker-main/bert_model'
tokenizer_path = '/initiativetracker-main/bert_model'

# Load the tokenizer and model from pre-trained or fine-tuned weights
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def predict_character_class(backstory):
    """
    Predict the character class from the input text using a fine-tuned BERT model.

    Parameters:
    text (str): Input text for which the character class needs to be predicted.

    Returns:
    str: Name of the predicted character class.
    """
    # Tokenize and encode the text as required by the BERT model
    inputs = tokenizer.encode_plus(
        backstory, 
        add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
        return_tensors="pt",       # Return PyTorch tensors
        truncation=True,            # Truncate to max model input length
        max_length=4512              # Maximum length for BERT inputs
    )

    # Make sure model is in evaluation mode
    model.eval()

    # No need to track gradients for prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # The outputs are logits, get the softmax to find the highest probability class
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the predicted class index
    #predicted_class_index = torch.argmax(probs, dim=-1).item()

    # Define a map from class indices to class names (update based on your classes)
    #class_names = {'Barbarian, Bard, Cleric, Druid, Fighter, Ranger, Rogue, Paladin, Warlock, Sorcerer, Wizard, Monk'}
    class_names = {0: 'Barbarian', 1: 'Bard', 2: 'Cleric', 3: 'Druid', 4: 'Fighter', 5: 'Ranger', 6: 'Rogue', 7: 'Paladin', 8: 'Warlock', 9: 'Sorcerer', 10: 'Wizard', 11: 'Monk'}

    # Return the predicted class name
    return class_names[torch.argmax(probs, dim=-1).item()]

# Testing the function to see if it's working correctly (This part is optional)
'''if __name__ == "__main__":
    # Sample text for testing the prediction
    test_backstory = "Character gains power from the elements and can control the weather."
    print(predict_character_class(test_backstory))  # Output the result of the prediction
'''