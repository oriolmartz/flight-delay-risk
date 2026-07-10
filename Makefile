.PHONY: setup test quality benchmark dashboard api docker

setup:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

quality:
	python -m scripts.quality_gate

benchmark:
	python -m scripts.benchmark_inference

dashboard:
	python -m streamlit run app/dashboard/streamlit_app.py

api:
	python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker compose up --build
