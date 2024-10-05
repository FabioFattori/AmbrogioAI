**AmbrogioAI**

Ambrogio is a Classifier Ai that get a array of 4096 elements, which are the extracted features from the image.
In the repo there is a feature extractor in the "classes" directory, but it can be subsituted by any feature extractor as long as a 4096 array is passed to the Neural Network.
Ambrogio is entirely written in python code with the help of the numpy library.

**Installation**

- Windows, Mac, Linux:
    
    first run the set up script which will download all the dependencies required by the AI to work, like this:
    
    On Mac/Linux:

    <code>./setUp.sh</code>
    
    On Windows:

    <code>.\setUp.bat</code>
    ___
    then if no errors appears you are good to go and run the start script like this:

    On Mac/Linux:

    <code>./start.sh</code>
    
    On Windows:

    <code>.\start.bat</code>

    *but if* you see some errors you have to install the dependencies manualy, which are list in <a href="script/dependencies.txt">this file</a>.
    Some times the errors are caused by the version of pip, the needing of a virtual enviroment, ecc.
    I suggest that you try with a virtual enviroment so that you can manage all the dependencies with much more control, try to follow the installation for the Raspberry to get more details about the virtual enviroment.
- Raspberry:

    start by creating a virtual enviroment for the dependencies of the project by running this command:

    <code>python3 -m venv "path that you desire for the directory of dependencies"</code>
    
    if you get an error is probably caused by the python version, try the same command with "python" instead of "python3"

    ___

    when you have created the virtual enviroment you have to run this command to activate it:

    <code>source "path of the created directory"</code>

    once the enviroment is activated you can follow the installation guide of Linux to install the dependencies and start the program

**Customization of the Dataset**

you can modify the labels by editing <a href="utilities/classes.txt">this file</a> which is a simple .txt where are listed the labels name, the associations between the imgs and the labels are in <a href="imgs/dataSet.json">this .json file</a>, where an array "imgs" is created in which every element is an object of label and img.
In the same folder of the .json file there are some dirs named after the labels and contains images based of the labels they are classified of.

**Versions**
- 1.00: 

    AmbrogioSimple, a version of the NN with a single hidden layer composed of n neurons (default = 64), is a simple NN, reliable but quite small without any type of optimizations add to it.

    &cross; ability to customize the structure of the NN by adding hidden layers.

    &cross; mini-batch support.

    &check; sistem which saves the state and the progress of the training and can load it back with every new iterations.

    &cross; presense of warm-up algorithms for the learning rate.

    &check; ability to customize the labels, expanding them or reducing them (this operation requires a new dataset created by the user).
