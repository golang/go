# Copyright 2018 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY: usage

usage:
	echo "See Makefile"
	exit 1

docker-prod: Dockerfile.prod
	cd ../..; docker build -f cmd/godoc/Dockerfile.prod --tag=gcr.io/golang-org/godoc:$(VERSION) .

push-prod: docker-prod
	docker push gcr.io/golang-org/godoc:$(VERSION)

deploy-prod: push-prod
	gcloud -q app deploy app.prod.yaml --project golang-org --no-promote --image-url gcr.io/golang-org/godoc:$(VERSION)

get-latest-url:
	@gcloud app versions list -s default --project golang-org --sort-by=~version.createTime --format='value(version.versionUrl)' --limit 1 | cut -f1

regtest:
	./regtest.bash $(shell make get-latest-url)
