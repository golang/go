# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker: Dockerfile
	rm -f docker-linux.base.tar.gz
	docker build -t gobuilders/linux-x86-base .

# This gets uploaded to:
# https://console.developers.google.com/project/apps~symbolic-datum-552/storage/go-builder-data/
docker-linux.base.tar.gz: docker
	docker save gobuilders/linux-x86-base | gzip > docker-linux.base.tar.gz
