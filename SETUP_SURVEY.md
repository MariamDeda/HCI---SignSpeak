# SignSpeak survey: setup (about 10 minutes, once)

Flow: **intro page (GitHub Pages) → Start → your Streamlit app → after 1 minute a green
"I'm done exploring" button → survey page (15 taps) → answers land in a Google Sheet.**

## 1. Copy the files into your project and push
Add or replace these in your project folder (keep your own config.py, models, data):
- `app.py`, `nav.py`, `theme.py`, `survey_mode.py`  (replace / add in the project root)
- the whole `docs/` folder (add)

Then in cmd:
    git add app.py nav.py theme.py survey_mode.py docs SETUP_SURVEY.md
    git commit -m "Add survey flow"
    git push
Streamlit redeploys by itself in a minute or two.

## 2. Create the Google Sheet that stores answers
1. Go to https://sheets.google.com, create a blank sheet, name it "SignSpeak survey".
2. Menu **Extensions → Apps Script**. Delete the sample code, paste all of `docs/apps_script.gs`, click Save.
3. Click **Deploy → New deployment**, gear icon → **Web app**.
   Execute as: **Me**. Who has access: **Anyone**. Click Deploy and allow the permissions
   ("Advanced → Go to project" if Google warns you; it is your own script).
4. Copy the **Web app URL** (ends with `/exec`).
5. Open `docs/config.js` and paste it into `SHEET_URL: "..."`. Save.
   Test: open that URL in a browser; it should say "SignSpeak survey endpoint is working".

## 3. Turn on GitHub Pages
GitHub repo → **Settings → Pages** → Source: **Deploy from a branch** → Branch: `main`, folder: **/docs** → Save.
After a minute your links are:
- **Share this with participants:** https://mariamdeda.github.io/HCI---SignSpeak/
- Survey page (the app links to it itself): https://mariamdeda.github.io/HCI---SignSpeak/survey.html

(If the page 404s, the repo must be public, or your plan must allow Pages on private repos.)

## 4. Test it yourself
Open the link, press Start, wait 60 s, press the green button, answer, and check a new row
appears in the sheet (tab "Responses"). Delete your test rows before sharing.

## Good to know
- Change the waiting time or the survey link in `survey_mode.py` (`WAIT_SECONDS`, `SURVEY_URL`).
- Change questions in `docs/survey.html` (the `Q = [...]` list). If you add or rename one, add its key to `COLUMNS` in `apps_script.gs` and **redeploy** (Deploy → Manage deployments → Edit → New version).
- If SHEET_URL is empty the survey still works, but only in "test mode" (answers are not saved anywhere).
- Sheet columns: ratings are 1-5 (5 = best). To summarise, click a column and look at the Sum/Average
  in the bottom-right of Google Sheets, or use `=AVERAGE(K2:K)`.
- Opening the app without `?survey=1` (your normal link) shows no timer or button.
