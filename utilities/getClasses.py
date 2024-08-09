
def getClasses():
    try:    
        # Get the classes from the file
        with open('utilities/classes.txt', 'r') as file:
            classes = file.readlines()
        # Remove the newline characters
        classes = [x.strip() for x in classes]
        return classes
    except Exception as e:
        raise e