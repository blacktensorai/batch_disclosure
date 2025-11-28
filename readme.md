### `README.md`

````markdown
# CatalystScan

CatalystScan is an AI-powered tool designed to scan and analyze company filings from the ASX and SEC. It identifies forward-looking disclosures and price-moving catalysts using NLP and OCR technology.

## Simplest Setup (Docker)

We have containerized the entire application. This means **you do not need to manually install** Python, Tesseract OCR, Poppler, or any libraries. Docker handles everything.

### Prerequisites
* **Docker Desktop**: Download and install it for [Windows](https://docs.docker.com/desktop/install/windows-install/) or [Mac](https://docs.docker.com/desktop/install/mac-install/).
    * *Note: Ensure Docker Desktop is running before proceeding.*

---

### Quick Start (3 Steps)

#### 1. Configuration
The tool needs your API key to function.
1.  Look for the file named `.env`.
2.  Open it in a text editor and paste your OpenAI API Key:
    ```ini
    OPENAI_API_KEY=sk-your-key-here (We have already placed the key here the one you us provided with)
    ```

#### 2. Run the Application
Open your terminal (Command Prompt or PowerShell on Windows) in the project folder and run:

```bash
docker-compose up --build
````

  * *This will download all dependencies, set up the OCR tools, and start the server. The first run may take a few minutes.*

#### 3\. Access the Dashboard

Once the terminal says the server is running, open your web browser and go to:
**http://localhost:8501**

-----

### Running the Scan Pipeline

The dashboard displays the data. To fetch **new** filings (run the scraper and AI analysis), open a **new** terminal window (leave the first one running) and use this command:

```bash
docker-compose exec catalystscan python pipeline/run_pipeline.py
```

-----

### Where is my data?

Even though the app runs in a container, your data is saved to your local machine.

  * **Database**: Saved in the `storage/` folder.
  * **PDFs**: Saved in `data/asx/` and `data/sec/`.
  * **Logs**: Saved in `logs/`.

*You can close Docker and restart it later without losing your scanned data.*

-----

### How to Stop

To stop the application, go to the terminal where the dashboard is running and press:
`Ctrl + C`

```