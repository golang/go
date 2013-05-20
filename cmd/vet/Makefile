# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Assumes go/types is installed
test testshort:
	go build
	$(GOROOT)/test/errchk ./vet -printfuncs='Warn:1,Warnf:1' test_*.go test_*.s

# Install command where the go tool can find it.
install:
	go build -o _vet
	cp _vet $(GOROOT)/pkg/tool/$(GOOS)_$(GOARCH)/vet
	rm -f _vet
