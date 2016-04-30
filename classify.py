import sys
from PIL import Image
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

# Number of machines created
NUM_MACHINES = 10

# Converts an image file to a list of black or white pixel values
# Arguments: the name of the file
def toPixelVector(filename):
    try:
        image = Image.open(sys.argv[1])
    except IOError:
        print("Please supply the name of a valid image file")
        exit()

    # Convert to grayscale, then black and white image
    image = image.convert("L")
    image = image.point(lambda px: 0 if px < 175 else 255, '1')
    data = list(image.getdata())
    
    return data

# Predicts the class for a given sample
# Arguments: a list of machine learning machines, and the sample vector
# to predict for
def predictClass(machines, sample):
    scoreSums = machines[0].predict_proba([sample])[0]
    highestCertainty = scoreSums.copy()
    
    for i in range(1, len(machines)):
        scores = machines[i].predict_proba([sample])[0]
        for i in range(0, len(scores)):
            scoreSums[i] += scores[i]
            # Keep track of the highest certainty score so far
            if (scores[i] > highestCertainty[i]):
                highestCertainty[i] = scores[i]
            
    # Find maximum
    highestIndex = 0
    for i in range(1, len(scoreSums)):
        if (scoreSums[i] > scoreSums[highestIndex]):
            highestIndex = i
        elif (scoreSums[i] == scoreSums[highestIndex] and \
        highestCertainty[i] > highestCertainty[highestIndex]):
            highestIndex = i
    
    # Class is index + 1
    return highestIndex + 1
    
def main():
    if (len(sys.argv) < 2):
        # Not enough arguments
        print("Please supply the name of an image file to classify")
    else:
        # Try to load the machines
        try:
            machines = list()
            for i in range(0, NUM_MACHINES):
                machines.append(joblib.load("machine" + str(i) + ".pkl"))
        except:
            print("A decision tree is missing. Please run training.py and try again")
            
        # Load the image file and classify with machines
        data = toPixelVector(sys.argv[1])
        classNum = predictClass(machines, data)
        
        # Use dictionary to map class number to name, and print out
        classes = {1: "Smile", 2: "Hat", 3:"Hash", 4:"Heart", 5:"Dollar"}
        print(classes[classNum])

main()
