.ONESHELL:
SHELL = /bin/bash

.PHONY: env
env :
	conda env update --name ligo --file environment.yml --prune

.PHONY: html
html :
	myst build --html

.PHONY: clean
clean :
	rm -rf _build/ audio/* figures/*