# Python Essentials for MLOps
In this repository, you will find resources and projects that cover essential Python skills for Machine Learning Operations (MLOps). These projects are based on Dataquest coursework and incorporate tools and techniques introduced in the "Machine Learning Based Systems Design" course.

## Project Descriptions
There are three projects in this repository:

- [Movie Recommendation System](https://github.com/gabrielaact/mlops/tree/main/Python%20Essentials%20for%20MLOps/Project%2001): The Movie Recommendation System is designed to provide movie recommendations based on user input. Users can enter the name of a movie, and the system generates recommendations using TF-IDF (Term Frequency-Inverse Document Frequency) to find similar movie titles. Additionally, users can input a movie ID to discover movies that are similar based on user ratings. The system utilizes Pandas for data manipulation, Scikit-learn for TF-IDF vectorization, and cosine similarity for recommendation calculations. To run the system, ensure you have Python 3.x, Pandas and Scikit-learn installed. The movie data is expected to be in CSV files, namely 'data/movies.csv' and 'data/ratings.csv'.
- [Airflow Data Pipeline to Download Podcasts](https://github.com/gabrielaact/mlops/tree/main/Python%20Essentials%20for%20MLOps/Project%2002): This project consists of an Apache Airflow Directed Acyclic Graph (DAG) that streamlines the process of summarizing podcast episodes. It retrieves podcast episodes from a specified URL, stores episode details in a SQLite database, downloads audio files, transcribes them into text, and stores the transcripts in the database. The system relies on several libraries, including requests for network requests, xmltodict for XML parsing, pydub for audio file processing, Vosk for speech recognition, and Apache Airflow for orchestrating the workflow. You can execute this DAG with Apache Airflow. Ensure you have Python 3.x, Apache Airflow, requests, xmltodict, pydub, and vosk installed. The SQLite database connection should be configured within your Apache Airflow environment.

## Certificate of Completion

Upon successfully completing the Dataquest course, a certificate of achievement is awarded. You can view and validate the certificate for the course 'Intermediate Python for Web Development' by following the link below:

[Dataquest Certificate - Intermediate Python for Web Development](https://app.dataquest.io/view_cert/MLO6Y4AP90Y4EETQXE7C)


## Requirements/Technologies

- Python 3.11
- Pandas
- Scikit-learn
- Airflow
- requests
- xmltodict
- pydub
- Vosk
- SQLite
- Pylint

## Installation Instructions


## Link to Video


## Certificate of Completion

Upon successfully completing the Dataquest course "Intermediate Python for Web Development," you can obtain a certificate of achievement. You can view and validate the certificate using the following link:

[Dataquest Certificate - Intermediate Python for Web Development](https://app.dataquest.io/view_cert/MLO6Y4AP90Y4EETQXE7C)


## References

- [Dataquest Project Solutions](https://github.com/dataquestio/solutions)

