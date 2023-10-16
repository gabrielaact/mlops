# Airflow Data Pipeline to Download Podcasts

This Python script defines an Apache Airflow Directed Acyclic Graph (DAG) for summarizing a podcast. It retrieves podcast episodes from a given URL, stores episode information in a SQLite database, downloads audio files, transcribes them into text, and stores the transcript in the database.

The script uses various libraries, including requests for network requests, xmltodict for XML parsing, pydub for audio file processing, Vosk for speech recognition, and Apache Airflow for workflow orchestration.

## Usage

1. **Requirements:**

   Make sure you have the following dependencies installed:
   - Python 3.x
   - Apache Airflow
   - requests
   - xmltodict
   - pydub
   - vosk

   The SQLite database connection is expected to be configured in an Apache Airflow environment.

2. **Running the DAG:**

   To execute the podcast summarization DAG with Apache Airflow, ensure you have Apache Airflow installed. You can trigger the DAG to run periodically.

3. **DAG Structure:**

   - **create_table_sqlite:** An operator that creates a SQLite table if it doesn't exist for storing episode details.
   - **get_episodes:** A task that retrieves episodes from the podcast feed and parses them.
   - **load_episodes:** A task that loads episodes into the SQLite database, updating the database with new episodes.
   - **download_episodes:** A task that downloads podcast audio files.
   - **speech_to_text:** A task that performs speech-to-text transcription on audio files and updates the database with transcripts.

4. **DAG Configuration:**

   - `PODCAST_URL`: URL of the podcast feed.
   - `EPISODE_FOLDER`: Directory where podcast episodes will be saved.
   - `FRAME_RATE`: Sample rate for speech recognition.

## Running the DAG

To run the DAG, you can use Apache Airflow. The DAG is configured to run daily, starting from the specified date.

## Linting and Pylint

Linting is the process of checking your code for potential issues, style violations, and errors. To ensure that the code follows good coding practices, we can use Pylint, a Python code linter, by running the following command:

```
pylint podcasts.py
```

The script movie_recommendation.py received a linting score of 9.03.

![Pylint result](https://github.com/gabrielaact/mlops/blob/main/Python%20Essentials%20for%20MLOps/Project%2001/images/pylint.png)

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html)
- [Dataquest - Build an Airflow Data Pipeline to Download Podcasts](https://github.com/dataquestio/project-walkthroughs/blob/master/podcast_summary/podcast_summary.py)

