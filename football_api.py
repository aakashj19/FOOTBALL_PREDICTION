
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")

def test_api_connection():
    if not API_KEY:
        print("API key not found. Check your .env file.")
        return

    url = "https://api.football-data.org/v4/competitions/PL"
    headers = {"X-Auth-Token": API_KEY}

    response = requests.get(url, headers=headers, timeout=15)

    if response.status_code == 200:
        data = response.json()
        print("API connection successful!")
        print("Competition:", data["name"])
        print("Season:", data["currentSeason"]["startDate"],
              "to", data["currentSeason"]["endDate"])
    else:
        print("API request failed:", response.status_code)
        print(response.text)


if __name__ == "__main__":
    test_api_connection()
