# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker: Dockerfile
	docker build -t gobuilders/linux-x86-sid .

docker-linux.sid.tar.gz: docker
	docker save gobuilders/linux-x86-sid | gzip | (cd ../../coordinator/buildongce && go run create.go --write_object=go-builder-data/docker-linux.sid.tar.gz)

check: docker
	docker run gobuilders/linux-x86-sid /usr/local/bin/builder -rev=8c27884843c3 -buildroot=/ -v -report=false linux-amd64-sid
