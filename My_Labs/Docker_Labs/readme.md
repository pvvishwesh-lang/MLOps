### ABOUT
In this Lab, we create a Docker Image, that when deployed, will read a csv, perform RandomizedSearchCV(Hyperparametertuning) and train the 
best performing model, save metrics and output along with the model. 

### INSTRUCTIONS

1: Clone the Github Repo

2: Navigate to My_Labs/Docker_Labs.

3: Open a terminal and run the following commands: 
   a: docker build -t diabetes-model . (or name it anything else)
   b: docker run -v $(pwd)/output:/app/output diabetes-model

4: Wait for a few minutes. Once the execution ends, a new folder, output, is created which contains all the outputs from the run

### REQUIREMENTS
requirements.txt contains all the required imports, if any new changes are needed, add the import to requirements.txt and deploy the image
