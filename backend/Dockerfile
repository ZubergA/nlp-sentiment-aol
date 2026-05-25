FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy medium model (lebih akurat dari sm)
RUN python -m spacy download en_core_web_md

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]