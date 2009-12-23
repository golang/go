# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.$(GOARCH)

TARG=container/vector
GOFILES=\
	defs.go\
	intvector.go\
	stringvector.go\
	vector.go\

generate: vector.go vector_test.go
	< vector.go cat\
	| gofmt -r='Vector -> IntVector'\
	| gofmt -r='interface{} -> int'\
	> intvector.go\
	
	< vector.go cat\
	| gofmt -r='Vector -> StringVector'\
	| gofmt -r='interface{} -> string'\
	> stringvector.go\
	
	< vector_test.go cat\
	| gofmt -r='Vector -> IntVector'\
	| gofmt -r='zero -> intzero'\
	| gofmt -r='elem2Value -> elem2IntValue'\
	| gofmt -r='intf2Value -> intf2IntValue'\
	| gofmt -r='int2Value -> int2IntValue'\
	| gofmt -r='TestZeroLen -> TestIntZeroLen'\
	| gofmt -r='TestResize -> TestIntResize'\
	| gofmt -r='TestResize2 -> TestIntResize2'\
	| gofmt -r='checkZero -> checkIntZero'\
	| gofmt -r='TestTrailingElements -> TestIntTrailingElements'\
	| gofmt -r='TestAccess -> TestIntAccess'\
	| gofmt -r='TestInsertDeleteClear -> TestIntInsertDeleteClear'\
	| gofmt -r='verify_slice -> verify_sliceInt'\
	| gofmt -r='verify_pattern -> verify_patternInt'\
	| gofmt -r='make_vector -> make_vectorInt'\
	| gofmt -r='TestInsertVector -> TestIntInsertVector'\
	| gofmt -r='TestDo -> TestIntDo'\
	| gofmt -r='TestIter -> TestIntIter'\
	| gofmt -r='TestVectorData -> TestIntVectorData'\
	> intvector_test.go\
	
	< vector_test.go cat\
	| gofmt -r='Vector -> StringVector'\
	| gofmt -r='zero -> strzero'\
	| gofmt -r='int2Value -> int2StrValue'\
	| gofmt -r='intf2Value -> intf2StrValue'\
	| gofmt -r='elem2Value -> elem2StrValue'\
	| gofmt -r='TestZeroLen -> TestStrZeroLen'\
	| gofmt -r='TestResize -> TestStrResize'\
	| gofmt -r='TestResize2 -> TestStrResize2'\
	| gofmt -r='checkZero -> checkStrZero'\
	| gofmt -r='TestTrailingElements -> TestStrTrailingElements'\
	| gofmt -r='TestAccess -> TestStrAccess'\
	| gofmt -r='TestInsertDeleteClear -> TestStrInsertDeleteClear'\
	| gofmt -r='verify_slice -> verify_sliceStr'\
	| gofmt -r='verify_pattern -> verify_patternStr'\
	| gofmt -r='make_vector -> make_vectorStr'\
	| gofmt -r='TestInsertVector -> TestStrInsertVector'\
	| gofmt -r='TestDo -> TestStrDo'\
	| gofmt -r='TestIter -> TestStrIter'\
	| gofmt -r='TestVectorData -> TestStrVectorData'\
	> stringvector_test.go

bench:
	gotest -v -match Nums -benchmarks Nums

include ../../../Make.pkg
