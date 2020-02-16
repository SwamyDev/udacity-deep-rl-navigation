.PHONY: help meta setup clean test coverage

PIP_OPTIONS ?= -e
PYTHON ?= python3

.DEFAULT: help
help:
	@echo "make meta"
	@echo "	update version number and meta data"
	@echo "make setup"
	@echo "	create virtual environment and install dependencies"
	@echo "make clean"
	@echo "	clean all python build/compilation files and directories"
	@echo "make test"
	@echo "	run all tests"
	@echo "make coverage"
	@echo " run all tests and produce coverage report"

meta:
	python meta.py `git describe --tags --abbrev=0`

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	rm --force .coverage
	rm --force --recursive .pytest_cache
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info
	rm --force resources/unity-mlagent/.coverage
	rm --force --recursive resources/unity-mlagent/.pytest_cache
	rm --force --recursive resources/unity-mlagent/build/
	rm --force --recursive resources/unity-mlagent/*.egg-info

resources/unity-mlagent/dist:
	test -d resources/unity-mlagent/venv || ${PYTHON} -m venv resources/unity-mlagent/venv
	cd resources/unity-mlagent; . venv/bin/activate; pip install --upgrade pip setuptools wheel
	cd resources/unity-mlagent; . venv/bin/activate;  python setup.py sdist bdist_wheel

resources/environments:
	resources/fetch-unity-environments.sh

venv/done: resources/unity-mlagent/dist resources/environments | clean
	test -d venv || ${PYTHON} -m venv venv
	. venv/bin/activate; pip install --upgrade pip
	. venv/bin/activate; pip install --upgrade setuptools
	. venv/bin/activate; pip install --find-links=resources/unity-mlagent/dist $(PIP_OPTIONS) .
	touch venv/done

setup: venv/done

venv/test_done: venv/done
	. venv/bin/activate; pip install --find-links=resources/unity-mlagent/dist $(PIP_OPTIONS) .[test]
	touch venv/test_done

test: venv/test_done
	. venv/bin/activate; pytest --verbose --color=yes tests 

coverage: venv/test_done
	. venv/bin/activate; pytest --cov=p1_navigation --cov-report term-missing tests
