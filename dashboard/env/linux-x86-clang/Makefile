# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker: Dockerfile
	docker build -t gobuilders/linux-x86-clang .

docker-linux.clang.tar.gz: docker
	docker save gobuilders/linux-x86-clang | gzip | (cd ../../coordinator/buildongce && go run create.go --write_object=go-builder-data/docker-linux.clang.tar.gz)

check: docker
	docker run gobuilders/linux-x86-clang /usr/local/bin/builder -rev=77e96c9208d0 -buildroot=/ -v -report=false linux-amd64-temp

check32: docker
	docker run gobuilders/linux-x86-clang /usr/local/bin/builder -rev=77e96c9208d0 -buildroot=/ -v -report=false linux-386-temp
