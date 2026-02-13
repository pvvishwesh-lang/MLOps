# APACHE BEAM LAB

## About
In this lab, I tried to connect a batch pipeline to an open source API to fetch Fifa World Cup 2026 Data,get the number of matches each group will be playing,
and store it locally in a txt file. 

## Working
As part of this project, I've created an account in a site called https://www.football-data.org/client/home, and received a token id that allows me to access the API for free.
I created a custom API connector class and pass a beam.DoFn object to it so Apache Beam can use it as a beam process and initialize the class and  define a process function,
which is required by Apache Beam in any custom connectors, and pass element as a parameter. Next, I set max_retries to 0 and then check if the response i get is 429, which means
I received a timeout from the API, so I then make the process sleep for a while and update the max_retires by 1. I check the API 5 times max and if I receive a timeout 5 times, 
I stop the process and raise an error, else I get the data from the API and convert it into a json format. Next, I define a function that will help me get the count per group.

Within the pipeline definition, I first create a Beam.Create(None) step, since Beam needs either a read or a create step as the first step of the pipeline, after which I then start 
reading from the API, flatten the data and aggregate it and get the count and in the end, save it as a txt file using the Beam.Writetotxt.

## Command to run the pipeline
python Get_Fifa_World_Cup_Data.py --runner=DirectRunner

## Requirements
Token to access the above mentioned site's API

