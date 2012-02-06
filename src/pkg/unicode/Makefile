# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

maketables: maketables.go
	go build maketables.go

tables:	maketables
	./maketables --tables=all > tables.go
	gofmt -w tables.go

# Downloads from www.unicode.org, so not part
# of standard test scripts.
testtables: maketables
	@echo '***' Be sure to make tables and go install first
	./maketables -test
