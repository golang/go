# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Assumes go/types is installed
test testshort:
	go build -tags 'vet_test gotypes'
	../../../test/errchk ./vet -printfuncs='Warn:1,Warnf:1' test_*.go test_*.s

test_notypes:
	go build -tags 'vet_test'
	# Only those tests that do not depend on types.
	# Excluded: test_print.go
	../../../test/errchk ./vet -printfuncs='Warn:1,Warnf:1' test_asm.go test_assign.go test_atomic.go test_buildtag.go test_buildtag_bad.go test_deadcode.go test_method.go test_rangeloop.go test_structtag.go test_taglit.go test_*.s
