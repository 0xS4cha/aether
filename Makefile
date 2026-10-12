.PHONY: install install-editable test sample run clean docker-build docker-run

install:
	uv sync

install-editable:
	uv pip install -e .

test:
	uv run pytest

sample:
	uv run aether generate-sample --output data/samples/sample_nebula.png

run:
	uv run aether visualize --sample

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf dist build *.egg-info

docker-build:
	docker build -t aether:latest .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data aether:latest visualize --sample --headless --screenshot data/outputs/docker_render.png
