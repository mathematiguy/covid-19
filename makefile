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

replicate:
	$(RUN) julia replicate.jl

PLUTO_PORT ?=1234
pluto: DOCKER_ARGS=-p $(PLUTO_PORT):$(PLUTO_PORT) -it
pluto:
	$(RUN) julia -e 'import Pluto; Pluto.run(host="0.0.0.0", require_secret_for_open_links=false, require_secret_for_access=false, port=$(PLUTO_PORT))'

r_shell: DOCKER_ARGS= -dit --rm -e DISPLAY=$$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --name="rdev"
r_shell:
	$(RUN) R

clean:
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
