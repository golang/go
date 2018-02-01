# Copyright 2017 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

VERSION=v2

.PHONY: usage

usage:
	echo "See Makefile"
	exit 1

update-deps:
	go install golang.org/x/build/cmd/gitlock
	gitlock --update=Dockerfile --ignore=NONE --tags=autocert golang.org/x/tools/cmd/tip

docker-prod: Dockerfile
	docker build -f Dockerfile --tag=gcr.io/symbolic-datum-552/tip:$(VERSION) .
docker-dev: Dockerfile
	docker build -f Dockerfile --tag=gcr.io/go-dashboard-dev/tip:$(VERSION) .

push-prod: docker-prod
	gcloud docker -- push gcr.io/symbolic-datum-552/tip:$(VERSION)
push-dev: docker-dev
	gcloud docker -- push gcr.io/go-dashboard-dev/tip:$(VERSION)
