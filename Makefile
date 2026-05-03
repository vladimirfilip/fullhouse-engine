.PHONY: install demo test validate clean

install:
	@echo ">> Installing Cython<3 (eval7 build dep)"
	pip3 install "Cython<3"
	@echo ">> Installing eval7 with --no-build-isolation"
	pip3 install --no-build-isolation eval7==0.1.7
	@echo ">> Installing rest of requirements"
	pip3 install flask numpy scipy treys scikit-learn

demo:
	python3 demo.py

test:
	python3 -m pytest tests/ -q

validate:
	@if [ -z "$(BOT)" ]; then echo "usage: make validate BOT=bots/mybot/bot.py"; exit 1; fi
	python3 sandbox/validator.py $(BOT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
