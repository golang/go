# Copyright 2018 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY: usage

GO_REF ?= release-branch.go1.11
TOOLS_HEAD := $(shell git rev-parse HEAD)
TOOLS_CLEAN := $(shell (git status --porcelain | grep -q .) && echo dirty || echo clean)
ifeq ($(TOOLS_CLEAN),clean)
	DOCKER_VERSION ?= $(TOOLS_HEAD)
else
	DOCKER_VERSION ?= $(TOOLS_HEAD)-dirty
endif
GCP_PROJECT := golang-org
DOCKER_TAG := gcr.io/$(GCP_PROJECT)/godoc:$(DOCKER_VERSION)

usage:
	@echo "See Makefile and README.godoc-app"
	@exit 1

docker: Dockerfile.prod
	# NOTE(cbro): move up in directory to include entire tools repo.
	cd ../..; docker build \
		-f cmd/godoc/Dockerfile.prod \
		--build-arg GO_REF=$(GO_REF) \
		--build-arg TOOLS_HEAD=$(TOOLS_HEAD) \
		--build-arg TOOLS_CLEAN=$(TOOLS_CLEAN) \
		--build-arg DOCKER_TAG=$(DOCKER_TAG) \
		--tag=$(DOCKER_TAG) \
		.

push: docker
	docker push $(DOCKER_TAG)

deploy: push
	gcloud -q app deploy app.prod.yaml \
		--project $(GCP_PROJECT) \
		--no-promote \
		--image-url $(DOCKER_TAG)

get-latest-url:
	@gcloud app versions list \
		-s default \
		--project $(GCP_PROJECT) \
		--sort-by=~version.createTime \
		--format='value(version.versionUrl)' \
		--limit 1 | cut -f1 # NOTE(cbro): gcloud prints out createTime as the second field.

regtest:
	go test -v \
		-regtest.host=$(shell make get-latest-url) \
		-run=Live
