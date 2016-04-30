from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn import cross_validation
from PIL import Image
import sys
import numpy

# Import a set of image data
# Arguments: the folder to import the images from, the class they're in,
# the image number to start at, and the number to import
def importData(folder, classNum, start, quantity):
    samples = list()
    classifications = list()
    
    for i in range(1, quantity + 1):
        filename = '{0}/{1:02}.jpg'.format(folder, i)
        
        image = Image.open(filename)

        # Convert to grayscale, then black and white image
        image = image.convert("L")
        image = image.point(lambda px: 0 if px < 175 else 255, '1')
        data = list(image.getdata())
        
        samples.append(data)
        classifications.append(classNum)
        
    return samples, classifications

# Chops up a data set into equal sized samples
# Arguments: a list of sample lists for each class, a list of class number
# lists, and the number of sets to divide the data into
def createDatasets(samples, classes, numSets):
    dataSets = list()
    dataClasses = list()    
    remainingData = samples.copy()
    remainingClasses = classes.copy()        
    
    for i in numpy.arange(1.0, 1 / numSets, -(1 / numSets)):
        for j in range(0, len(samples)):
            # Create random partition of data
            tempData, remainingData[j], tempClasses, remainingClasses[j] = \
            cross_validation.train_test_split(remainingData[j], remainingClasses[j], test_size=1.0 - 0.1 / i, random_state=100) 
    
            if j == 0:
                dataSets.append(tempData)
                dataClasses.append(tempClasses)
            else:
                dataSets[-1].extend(tempData)
                dataClasses[-1].extend(tempClasses)
        print(i, int(round(numSets - i*numSets)), len(dataSets[-1]))
            
    # Use remaining as last data set
    dataSets.append(remainingData[0])
    dataClasses.append(remainingClasses[0])
    for i in range(1, len(remainingData)):
        dataSets[-1].extend(remainingData[i])
        dataClasses[-1].extend(remainingClasses[i])
    print(1 / numSets, len(dataSets) - 1, len(dataSets[-1]))

    return dataSets, dataClasses
    
def main():
    # Create samples and class vectors
    samples = list()
    classes = list()

    if (len(sys.argv) > 1):
        # Number of machines was provided
        numMachines = int(sys.argv[1])
    else:
        # Default is 10
        numMachines = 10
        
    if (len(sys.argv) > 2):
        # Data path was provided
        dataPath = sys.argv[2]
        # Put slash on end, if need-be
        if (dataPath[-1] != '/'):
            dataPath += '/'
    else:
        dataPath = ''
    
    print("Loading samples...")

    # List of tuples containing data samples
    dataset = [('data/01', 1, 1, 85), ('data/02', 2, 1, 72), \
               ('data/03', 3, 1, 88), ('data/04', 4, 1, 81), \
               ('data/05', 5, 1, 87)]
    
    # Load samples data from image files
    for classType in dataset:
        print("\tLoading samples for class", classType[1])
        tempSamples, tempClasses = importData(dataPath + classType[0], classType[1], \
        classType[2], classType[3])

        samples.append(tempSamples)    
        classes.append(tempClasses)

    print("Done loading samples")
    
    # Divide data up into a set for each machine
    print("")
    print("Creating", numMachines, "datasets")
    dataSets, dataClasses = createDatasets(samples, classes, 10)
    print("Done creating datasets")

    # Train up machines    
    print("")
    print("Training machines")        
    machines = list()
    for i in range(0, len(dataSets)):
        tempMachine = DecisionTreeClassifier(criterion='entropy', random_state=100)
        tempMachine.fit(dataSets[i], dataClasses[i])
        machines.append(tempMachine)
        
    print("Done training machines")
    
    print("")
    print("Saving machines to file")
    for i in range(0, len(machines)):
        joblib.dump(machines[i], "machine" + str(i) + ".pkl")
    print("Machines saved")
main()