cd backend
docker build -t app_backend

cd ../frontend
docker build -t app_frontend

docker run --gpus=all -d -p 8080:8080 app_backend
docker run -d -p 8501:8501 app_frontend