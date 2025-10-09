# Copyright (C) 2025 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import csv
import re
import time
from datetime import timedelta

# from tqdm import tqdm
# from youtubesearchpython import VideosSearch


def parse_duration(duration_str: str):
    # Split the string by the colon
    parts = duration_str.split(":")

    # Initialize hours, minutes, and seconds
    hours = 0
    minutes = 0
    seconds = 0

    # Depending on the number of parts, assign values
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
    elif len(parts) == 1:
        seconds = int(parts[0])
    else:
        raise ValueError("Invalid duration format")

    # Create a timedelta object
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


class Video:
    def __init__(self, result: dict):
        self.title = result["title"]
        self.link = result["link"]
        self.id = result["id"]
        self.duration = parse_duration(result["duration"]) if result["duration"] else timedelta(0)
        if result["viewCount"]["text"]:
            view_count = result["viewCount"]["text"].replace(",", "")
            view_count = re.search(r"(\d+)", view_count)
            self.view_count = int(view_count.group(1)) if view_count else 0
        else:
            self.view_count = 0

    def __str__(self):
        return f"{self.title} [{self.id}] ({self.duration}) - {self.view_count} views"

    def __eq__(self, other):
        if isinstance(other, Video):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


def search_for_videos(query: str, limit: int = 100, keywords: list[str] = []):
    results = []
    pbar = tqdm(total=limit)
    search = VideosSearch(query, limit=50)
    result = search.result()["result"]
    pbar.update(len(result))
    results += result
    while len(results) < limit:
        time.sleep(0.3)
        if search.next():
            result = search.result()["result"]
            if len(result) == 0:
                break
            pbar.update(len(result))
            results += result
        else:
            break
    videos = [Video(r) for r in results if exclude_keywords(r["title"], keywords)]
    return videos


def exclude_keywords(video_or_title: Video | str, keywords: list[str]):
    if isinstance(video_or_title, Video):
        video_or_title = video_or_title.title
    return all(keyword not in video_or_title.lower() for keyword in keywords)


def write_to_csv(videos: list[Video], filename: str):
    # write to csv with csv package
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "link", "id", "duration", "view_count"])
        for video in videos:
            writer.writerow([video.title, video.link, video.id, video.duration, video.view_count])
    print(f"Saved {len(videos)} videos to {filename}")


def main(query, filename, keywords):

    results: list[Video] = []
    for q in query:
        try:
            results += search_for_videos(q, 200, keywords)
            print(len(results))
        except Exception as e:
            print(f"Error searching for {q}: {e}")
            continue

    # Unique results
    results = list(set(results))
    results = sorted(results, key=lambda v: v.view_count, reverse=True)

    write_to_csv(results, filename)


if __name__ == "__main__":

    # piano
    query_piano = [
        "钢琴独奏",
        "钢琴独奏曲",
        "piano solo",
        "ピアノソロ",
        "피아노 솔로",
        "Solo de piano",
        "Solo de piano",  # 西班牙语
        "Klaviersolo",  # 德语
        "Assolo di pianoforte",  # 意大利语
        "โซโล่เปียโน",  # 泰语
    ]
    filename = "piano_solo.csv"

    # Acoustic guitar
    # query_acoustic_guitar = [
    #     "原声吉他独奏",
    #     "木吉他独奏",
    #     "acoustic guitar solo",
    #     "アコースティックギターソロ",
    #     "어쿠스틱 기타 솔로",
    #     "Solo de guitare acoustique",
    #     "Solo de guitarra acústica",
    #     "Akustikgitarren-Solo",
    #     "Solo di chitarra acustica",
    #     "โซโล่กีตาร์แอคูสติก",
    # ]

    # Electric guitar, Bass
    # query_electric_guiatr = [
    #     "电吉他独奏",
    #     "エレキギターソロ",
    #     "electric guitar solo" "일렉트릭 기타 솔로",
    #     "Solo de guitare électrique",
    #     "Solo de guitarra eléctrica",
    #     "Elektrogitarren-Solo",
    #     "Solo di chitarra elettrica",
    #     "โซโล่กีตาร์ไฟฟ้า",
    # ]

    # Strings: violin, viola, cello, doublebass
    # query_strings = [
    #     "小提琴独奏",
    #     "小提琴独奏曲",
    #     "violin solo",
    #     "バイオリンソロ",
    #     "바이올린 솔로",
    #     "solo de violín",
    #     "Violinsolo",
    #     "Assolo di violino",
    #     "โซโล่ไวโอลิน",
    #     "скрипка соло"
    # ]

    # Wind-brass： horn，french-horn，euphonium，tuba，Trumpet
    # query_wind_brass = [
    #     "圆号独奏",
    #     "horn solo",
    #     "Horn-Solo",
    #     "Solo de corno",
    #     "ソロ・ホルン",
    #     "호른 솔로",
    #     "Assolo di corno",
    #     "เดี่ยวฮอร์น",
    #     "Cor solo",
    #     "соло на валторне"
    # ]

    # Wind-reeds： Englishhorn，bassoon，clarinet，contrabassoon，flute，oboe，piccolo，saxophone
    # query_wind_reeds = [
    #     "萨克斯独奏",
    #     "saxophone solo",
    #     "サックスソロ",
    #     "색소폰 솔로",
    #     "Solo de saxofón",
    #     "Saxophon-Solo",
    #     "Assolo di sassofono",
    #     "โซโล่แซกโซโฟน",
    #     "сольо на саксофоне",
    #     "saxo solo"
    # ]

    # Drums
    # query_drums = [
    #     "鼓独奏",
    #     "鼓独奏曲",
    #     "drum solo",
    #     "ドラムソロ",
    #     "드럼 솔로",
    #     "solo de batería",
    #     "Schlagzeugsolo",
    #     "assolo di batteria",
    #     "เดี่ยวกลอง",
    # ]

    filename = "piano_solo.csv"
    main(query_piano, filename, exclude_keywords=["string", "drums", "wind", "brass", "guiatr"])
