# Copyright 2017 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

VERSION ?= $(shell git rev-parse --short HEAD)
MUTABLE_VERSION ?= latest

IMAGE_STAGING := gcr.io/go-dashboard-dev/tip
IMAGE_PROD := gcr.io/symbolic-datum-552/tip

.PHONY: usage

usage:
	echo "See Makefile"
	exit 1

update-deps:
	go install golang.org/x/build/cmd/gitlock
	gitlock --update=Dockerfile --ignore=NONE --tags=autocert golang.org/x/tools/cmd/tip

docker-image: Dockerfile *.go
	docker build --force-rm -f Dockerfile --tag=$(IMAGE_PROD):$(VERSION) .
	docker tag $(IMAGE_PROD):$(VERSION) $(IMAGE_PROD):$(MUTABLE_VERSION)
	docker tag $(IMAGE_PROD):$(VERSION) $(IMAGE_STAGING):$(VERSION)
	docker tag $(IMAGE_PROD):$(VERSION) $(IMAGE_STAGING):$(MUTABLE_VERSION)

push-prod: docker-image
	docker push $(IMAGE_PROD):$(MUTABLE_VERSION)
	docker push $(IMAGE_PROD):$(VERSION)
push-staging: docker-image
	docker push $(IMAGE_STAGING):$(MUTABLE_VERSION)
	docker push $(IMAGE_STAGING):$(VERSION)

deploy-prod: push-prod
	go install golang.org/x/build/cmd/xb
	xb --prod kubectl set image deployment/tip-deployment tip=$(IMAGE_PROD):$(VERSION)
deploy-staging: push-staging
	go install golang.org/x/build/cmd/xb
	xb --staging kubectl set image deployment/tip-deployment tip=$(IMAGE_STAGING):$(VERSION)

