FROM anibali/pytorch:2.0.0-cuda11.8

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "main.py"]