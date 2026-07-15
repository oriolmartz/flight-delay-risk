.PHONY: setup setup-advanced prepare schedule-context ablation train backtest stability policy-backtest layer4 test neural-smoke quality benchmark dashboard api docker

setup:
	python -m pip install -r requirements.txt

setup-advanced:
	python -m pip install -r requirements-advanced.txt

prepare:
	python -m scripts.prepare_data

schedule-context:
	python -m scripts.build_schedule_context

ablation:
	python -m scripts.run_feature_ablation --max-rows 30000

train:
	python -m scripts.train_model --max-rows 30000 --candidate-profile flagship

backtest:
	python -m scripts.run_temporal_backtest --max-rows 9000 --n-splits 3 --candidate-profile flagship

stability:
	python -m scripts.run_feature_stability --max-rows 30000 --n-splits 3

policy-backtest:
	python -m scripts.run_policy_backtest --max-rows 30000 --n-splits 3

layer4:
	python -m scripts.build_layer4_release

test:
	python -m pytest -q

neural-smoke:
	python -m scripts.neural_smoke

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
