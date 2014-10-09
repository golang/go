# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker: Dockerfile
	docker build -t gobuilders/linux-x86-gccgo .

docker-linux.gccgo.tar.gz: docker
	docker save gobuilders/linux-x86-gccgo | gzip | (cd ../../coordinator/buildongce && go run create.go --write_object=go-builder-data/docker-linux.gccgo.tar.gz)

check: docker
	docker run gobuilders/linux-x86-gccgo /usr/local/bin/builder -tool="gccgo" -rev=b9151e911a54 -v -cmd='make RUNTESTFLAGS="--target_board=unix/-m64" check-go' -report=false linux-amd64-gccgo-temp

check32: docker
	docker run gobuilders/linux-x86-gccgo /usr/local/bin/builder -tool="gccgo" -rev=b9151e911a54 -v -cmd='make RUNTESTFLAGS="--target_board=unix/-m32" check-go' -report=false linux-386-gccgo-temp
