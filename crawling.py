import requests
from bs4 import BeautifulSoup
import pandas as pd

df = pd.read_csv("movies.csv")

data = {'movieId': [], 'director': []}

title_column = df['title']
movieId_column = df['movieId']

main_url = "https://imdb.com"

for i in range(3247, 6494):
    print(movieId_column[i], title_column[i],  "ongoing...")
    movie_name = title_column[i]

    search_url = "https://www.imdb.com/find/?q={}&ref_=nv_sr_sm".format(movie_name)
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers, timeout = 100)

    if response.status_code == 200:
        html = response.text
    else:
        print("Error! failed to bring page")
        continue

    soup = BeautifulSoup(html, "html.parser")
    try:
        # 가져온 html에서 첫 번째 감독, 배우 부분 찾아서 가져오기
        first_result = soup.find("a", class_="ipc-metadata-list-summary-item__t")
        if first_result:
            relative_link = first_result["href"]
            full_url = main_url + relative_link
            response = requests.get(full_url, headers=headers, timeout = 100)
            if response.status_code == 200:
                html = response.text
            else:
                print("no link found")
                continue
        soup = BeautifulSoup(html, "html.parser")
        info = soup.find("a", class_="ipc-metadata-list-item__list-content-item").get_text()
    except AttributeError:
        print(f"Warning: Text not found for movie {movie_name}")
        info = "Not available"

    data['movieId'].append(movieId_column[i])
    data['director'].append(info)

output_df = pd.DataFrame(data)

output_df.to_csv('output.csv', index=False)
