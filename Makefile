.PHONY: help meta setup clean test

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

resources/unity-mlagent/dist:
	test -d resources/unity-mlagent/venv || python3.7 -m venv resources/unity-mlagent/venv
	cd resources/unity-mlagent; ls 
	cd resources/unity-mlagent; . venv/bin/activate; pip install --upgrade pip setuptools wheel
	cd resources/unity-mlagent; . venv/bin/activate;  python setup.py sdist bdist_wheel

venv/done: resources/unity-mlagent/dist
	test -d venv || python3.7 -m venv venv
	. venv/bin/activate; pip install --upgrade pip
	. venv/bin/activate; pip install --upgrade setuptools
	. venv/bin/activate; pip install --find-links=resources/unity-mlagent/dist .
	touch venv/done

venv: venv/done

setup: meta clean | venv

venv/test_done: | setup
	. venv/bin/activate; pip install .[test]
	touch venv/test_done

test: | venv/test_done
	. venv/bin/activate; pytest --verbose --color=yes tests 

