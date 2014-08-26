# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker: Dockerfile
	docker build -t gobuilders/linux-x86-base .

push: docker
	docker push gobuilders/linux-x86-base
