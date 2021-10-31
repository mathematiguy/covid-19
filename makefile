REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
GIT_TAG ?= $(shell git log --oneline | head -n1 | awk '{print $$1}')
DOCKER_REGISTRY := mathematiguy
IMAGE := $(DOCKER_REGISTRY)/$(REPO_NAME)
HAS_DOCKER ?= $(shell which docker)
RUN ?= $(if $(HAS_DOCKER), docker run $(DOCKER_ARGS) --rm -v $$(pwd):/home/kaimahi/$(REPO_NAME) -w /home/kaimahi/$(REPO_NAME) -u $(UID):$(GID) $(IMAGE))
UID ?= kaimahi
GID ?= kaimahi
DOCKER_ARGS ?=

.PHONY: docker docker-push docker-pull enter enter-root

stan/sir_model.rds: R/fit_stan_model.R data/auckland_cases.csv stan/sir_model.stan
	$(RUN) Rscript $< \
		--case_data data/auckland_cases.csv \
		--model_code stan/sir_model.stan \
		--model_path $@

data/auckland_cases.csv: R/load_auckland_case_data.R nz-covid19-data-auto/cases_by_DHB_over_time.csv
	$(RUN) Rscript $< \
		--case_data nz-covid19-data-auto/cases_by_DHB_over_time.csv \
		--output $@ \
		--credentials credentials.yaml

replicate:
	$(RUN) julia replicate.jl

update: nz-covid19-data-auto/cases_by_DHB_over_time.csv
nz-covid19-data-auto/cases_by_DHB_over_time.csv:
	(cd nz-covid19-data-auto && git pull)
	$(RUN) bash -c 'cd nz-covid19-data-auto && python3 build_cases_by_dhb_over_time.py'

report: _book/_main.html
_book/_main.html: report/index.Rmd stan/sir_model.rds
	$(RUN) Rscript -e 'bookdown::render_book("$<")'

JUPYTER_PASSWORD ?= jupyter
JUPYTER_PORT ?= 8888
.PHONY: jupyter
jupyter: UID=root
jupyter: GID=root
jupyter: DOCKER_ARGS=-u $(UID):$(GID) --rm -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -e NB_USER=$$USER -e NB_UID=$(UID) -e NB_GID=$(GID)
jupyter:
	$(RUN) jupyter lab \
		--allow-root \
		--port $(JUPYTER_PORT) \
		--ip 0.0.0.0 \
		--NotebookApp.password=$(shell $(RUN) \
			python3 -c \
			"from IPython.lib import passwd; print(passwd('$(JUPYTER_PASSWORD)'))")

PLUTO_PORT ?=1234
pluto: DOCKER_ARGS=-p $(PLUTO_PORT):$(PLUTO_PORT) -it
pluto:
	$(RUN) julia -e 'import Pluto; Pluto.run(host="0.0.0.0", require_secret_for_open_links=false, require_secret_for_access=false, port=$(PLUTO_PORT))'

r_shell: DOCKER_ARGS= -dit --rm -e DISPLAY=$$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --name="rdev"
r_shell:
	$(RUN) R

clean:
	rm -f _main.Rmd stan/sir_model.rds
	find . -name '*backup*.jl' | xargs -I{} rm "{}"

docker:
	docker build $(DOCKER_ARGS) --tag $(IMAGE):$(GIT_TAG) .
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

docker-push:
	docker push $(IMAGE):$(GIT_TAG)
	docker push $(IMAGE):latest

docker-pull:
	docker pull $(IMAGE):$(GIT_TAG)
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

enter: DOCKER_ARGS=-it
enter:
	$(RUN) bash

enter-root: DOCKER_ARGS=-it
enter-root: UID=root
enter-root: GID=root
enter-root:
	$(RUN) bash
