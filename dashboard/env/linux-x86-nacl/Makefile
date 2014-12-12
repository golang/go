# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker: Dockerfile
	docker build -t gobuilders/linux-x86-nacl .

upload: docker
	docker save gobuilders/linux-x86-nacl | gzip | (cd ../../coordinator/buildongce && go run create.go --write_object=go-builder-data/docker-linux.nacl.tar.gz)

check: docker
	docker run gobuilders/linux-x86-nacl /usr/local/bin/builder -rev=77e96c9208d0 -buildroot=/ -v -cmd=/usr/local/bin/build-command.pl -report=false nacl-amd64p32
