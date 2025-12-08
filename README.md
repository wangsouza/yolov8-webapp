YOLOv8 WebApp (FastAPI + Frontend)

Aplicação web para detecção em tempo real via câmera, com controles de inferência (confiança/threshold e seleção de classes) e área de logs. Inclui CI com GitHub Actions (build/test/lint + Docker image) e CD com Helm Chart e ArgoCD.

Requisitos:
- Python 3.12
- Docker

Executar localmente:
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Build Docker:
docker build -t yolov8-webapp:latest .
docker run --rm -p 8000:8000 yolov8-webapp:latest

CI (GitHub Actions):
- Workflow: .github/workflows/ci.yml faz lint, testes básicos e build da imagem Docker.

CD (Helm + ArgoCD):
- Helm chart: deploy/helm/yolov8-webapp
- ArgoCD App: deploy/argocd/app.yaml

Instalar com Helm:
helm upgrade --install yolov8-webapp deploy/helm/yolov8-webapp --namespace yolov8 --create-namespace

ArgoCD:
- Aplique deploy/argocd/app.yaml no cluster com ArgoCD instalado e apontando para seu repositório Git.

Observações:
- Inferência roda em CPU por padrão (Ultralytics YOLOv8).
