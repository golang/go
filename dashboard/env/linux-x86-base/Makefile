# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker: Dockerfile
	docker build -t gobuilders/linux-x86-base .

docker-linux.base.tar.gz: docker
	docker save gobuilders/linux-x86-base | gzip | (cd ../../coordinator/buildongce && go run create.go --write_object=go-builder-data/docker-linux.base.tar.gz)

check: docker
	docker run gobuilders/linux-x86-base /usr/local/bin/builder -rev=3fa5d3a6dc9 -buildroot=/ -v -report=false linux-amd64-temp
