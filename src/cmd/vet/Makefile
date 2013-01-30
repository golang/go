# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

test testshort:
	go build
	../../../test/errchk ./vet -printfuncs='Warn:1,Warnf:1' print.go rangeloop.go atomic.go
