# Copyright 2017 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

VERSION=v1

docker-prod: Dockerfile
	docker build -f Dockerfile --tag=gcr.io/symbolic-datum-552/tip:$(VERSION) .
docker-dev: Dockerfile
	docker build -f Dockerfile --tag=gcr.io/go-dashboard-dev/tip:$(VERSION) .

push-prod: docker-prod
	gcloud docker push -- gcr.io/symbolic-datum-552/tip:$(VERSION)
push-dev: docker-dev
	gcloud docker push -- gcr.io/go-dashboard-dev/tip:$(VERSION)
