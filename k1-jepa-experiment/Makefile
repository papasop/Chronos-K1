.PHONY: baseline k1_v1 compare

baseline:
	python -m src.training.train_baseline

k1_v1:
	python -m src.training.train_k1_hybrid

compare:
	python -m src.eval.compare_baseline_vs_k1
