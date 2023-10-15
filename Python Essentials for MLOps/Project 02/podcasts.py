"""
Podcast Summary DAG

This Python script defines an Apache Airflow Directed Acyclic Graph (DAG) for 
summarizing a podcast. It retrieves podcast episodes from a given URL, stores 
episode information in a SQLite database, downloads audio files, transcribes them 
into text, and stores the transcript in the database.

The script uses various libraries, including requests for network requests, xmltodict 
for XML parsing, pydub for audio file processing, Vosk for speech recognition, and 
Apache Airflow for workflow orchestration.

Usage:
- Run this script with Apache Airflow to execute the podcast summarization DAG.

Requirements:
- Python 3.x
- Apache Airflow
- requests
- xmltodict
- pydub
- vosk

The SQLite database connection is expected to be configured in an 
Apache Airflow environment.

"""

# Import libraries
import json
import logging
import os
import pendulum
import requests
import xmltodict

from airflow.decorators import dag, task
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.providers.sqlite.operators.sqlite import SqliteOperator

from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO)

# URL of the podcast feed
PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"

# Directory where podcast episodes will be saved
EPISODE_FOLDER = "episodes"

# Sample rate for speech recognition
FRAME_RATE = 16000

@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",  # Scheduled to run daily
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False  # Do not catch up on missed tasks
)
def podcast_summary():
    """
    Declare the DAG.

    Args:
    - None.

    Returns:
    - None.
    """

    # Define an operator to create a SQLite table if it doesn't exist
    create_database = SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

    @task()
    def get_episodes():
        """
        Retrieve episodes from the podcast feed.

        Returns:
        - episodes: List of episodes in XML format converted to 
        dictionaries.
        """
        try:
            data = requests.get(PODCAST_URL, timeout=10)
            feed = xmltodict.parse(data.text)
            episodes = feed["rss"]["channel"]["item"]
            logging.info("Found %s episodes.", len(episodes))
            return episodes
        except requests.RequestException as e:
            logging.error("Error while retrieving episodes: %s", e)
            raise

    podcast_episodes = get_episodes()
    create_database.set_downstream(podcast_episodes)

    @task()
    def load_episodes(episodes):
        """
        Load episodes into the SQLite database if not already present.

        Args:
        - episodes: List of episodes.

        Returns:
        - new_episodes: List of episodes that were loaded into the database.
        """
        try:
            hook = SqliteHook(sqlite_conn_id="podcasts")
            stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
            new_episodes = []
            for episode in episodes:
                if episode["link"] not in stored_episodes["link"].values:
                    filename = f"{episode['link'].split('/')[-1]}.mp3"
                    new_episodes.append([episode["link"],
                                         episode["title"],
                                         episode["pubDate"],
                                         episode["description"],
                                         filename])

            hook.insert_rows(table='episodes',
                             rows=new_episodes,
                             target_fields=["link",
                                            "title",
                                            "published",
                                            "description",
                                            "filename"])
            logging.info("Loaded %s new episodes into the database.", len(new_episodes))
            return new_episodes
        except Exception as e:
            logging.error("Error while loading episodes into the database: %s", e)
            raise

    new_episodes = load_episodes(podcast_episodes)

    @task()
    def download_episodes(episodes):
        """
        Download podcast audio files.

        Args:
        - episodes: List of podcast episodes.

        Returns:
        - audio_files: List of downloaded audio files with episode links 
        and filenames.
        """
        audio_files = []
        for episode in episodes:
            try:
                name_end = episode["link"].split('/')[-1]
                filename = f"{name_end}.mp3"
                audio_path = os.path.join(EPISODE_FOLDER, filename)
                if not os.path.exists(audio_path):
                    logging.info("Downloading %s", filename)
                    audio = requests.get(episode["enclosure"]["@url"], timeout=10)
                    audio.raise_for_status()
                    with open(audio_path, "wb+") as f:
                        f.write(audio.content)
                audio_files.append({
                    "link": episode["link"],
                    "filename": filename
                })
            except (requests.RequestException) as e:
                logging.error("Error while downloading episodes: %s", e)
                raise
        return audio_files

    audio_files = download_episodes(podcast_episodes)

    @task()
    def speech_to_text(audio_files, new_episodes):
        """
        Perform speech-to-text transcription on audio files.

        Args:
        - audio_files: List of audio files to transcribe.
        - new_episodes: List of new episodes to update with transcripts.

        The function transcribes audio files and updates the database with 
        transcripts.

        Returns:
        - None
        """
        hook = SqliteHook(sqlite_conn_id="podcasts")
        untranscribed_episodes = hook.get_pandas_df(
            "SELECT * from episodes WHERE transcript IS NULL;")

        model = Model(model_name="vosk-model-en-us-0.22-lgraph")
        rec = KaldiRecognizer(model, FRAME_RATE)
        rec.SetWords(True)

        for _, row in untranscribed_episodes.iterrows():
            try:
                logging.info("Transcribing %s", row['filename'])
                filepath = os.path.join(EPISODE_FOLDER, row["filename"])
                mp3 = AudioSegment.from_mp3(filepath)
                mp3 = mp3.set_channels(1)
                mp3 = mp3.set_frame_rate(FRAME_RATE)

                step = 20000
                transcript = ""
                for i in range(0, len(mp3), step):
                    logging.info("Progress: %s", i/len(mp3))
                    segment = mp3[i:i+step]
                    rec.AcceptWaveform(segment.raw_data)
                    result = rec.Result()
                    text = json.loads(result)["text"]
                    transcript += text
                hook.insert_rows(table='episodes', rows=[[row["link"],
                                                          transcript]],
                                                          target_fields=["link",
                                                                         "transcript"],
                                                          replace=True)
            except Exception as e:
                logging.error("Error while transcribing episodes: %s", e)
                raise

    speech_to_text(audio_files, new_episodes)

if __name__ == '__main__':
    SUMMARY = podcast_summary()
