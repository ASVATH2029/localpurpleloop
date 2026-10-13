# Purple Loop - one-command entry points (Linux/macOS/Git Bash).
# Windows PowerShell users: see run.ps1

PY ?= ./.venv/Scripts/python.exe
ROUNDS ?= 20

.PHONY: help venv test run report registry ablate reset docker docker-run

help:
	@echo "make venv     - create .venv and install requirements"
	@echo "make test     - run the test suite"
	@echo "make run      - run the loop for ROUNDS rounds (default 20)"
	@echo "make report   - rebuild figures and print the summary"
	@echo "make registry - print the rule registry audit trail"
	@echo "make ablate   - compare feature groups (delivery vs technique)"
	@echo "make reset    - delete data/, logs/, ablation/ and figures"
	@echo "make docker-run - run the loop in a no-network container"

venv:
	python -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

run:
	$(PY) -m purpleloop run --rounds $(ROUNDS)

report:
	$(PY) -m purpleloop report

registry:
	$(PY) -m purpleloop registry

ablate:
	$(PY) -m purpleloop ablate --rounds 12

reset:
	$(PY) -m purpleloop reset --yes

docker:
	docker build -t purple-loop .

docker-run: docker
	docker run --rm --network none purple-loop
