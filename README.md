# A projekt témája

- Title és descriptionból valamilyen chart (mondjuk piechart vagy barchart), meg egy input, ahova ha beír az ember egy adott szót akkor megmutatja hogy melyik oldalon milyen gyakran fordul elő az adott szó (pl hányszor itt és hányszor ott)

- Emellett lehetne egy táblázat is, ami mutatja az egyes oldalak top 10 leggyakoribb szavait (stopwordök nélkül)

# A projekt futtatása

1. Klónozd a repót a gépedre.

2. Telepítsd a szükséges csomagokat a `requirements.txt` fájl alapján:

   ```bash
   pip install -r requirements.txt
   ```

2. Futtasd a `fetch_data.py` fájlt az adatok lekéréséhez és elmentéséhez:

   ```bash
   python fetch_data.py
   ```

3. Futtasd a `analyze_data.py` fájlt a webalkalmazás elindításához:

   ```bash
   python -m panel serve analyze_data.py --show
    ```
    
# Adatforrások

Telex: https://telex.hu/rss
Blikk: https://www.blikk.hu/rss/
Portfolio: https://www.portfolio.hu/rss/all.xml
24: https://24.hu/feed/
hvg: https://hvg.hu/rss
444: https://444.hu/feed


<img width="576" height="324" alt="preUI" src="https://github.com/user-attachments/assets/4cb5bfb5-b186-4895-9140-ee7431df8b43" />
