# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.$(GOARCH)

TARG=exp/vector
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
	| gofmt -r='TestZeroLenExp -> TestIntZeroLenExp'\
	| gofmt -r='TestResizeExp -> TestIntResizeExp'\
	| gofmt -r='TestResize2Exp -> TestIntResize2Exp'\
	| gofmt -r='checkZeroExp -> checkIntZeroExp'\
	| gofmt -r='TestTrailingElementsExp -> TestIntTrailingElementsExp'\
	| gofmt -r='TestAccessExp -> TestIntAccessExp'\
	| gofmt -r='TestInsertDeleteClearExp -> TestIntInsertDeleteClearExp'\
	| gofmt -r='verify_sliceExp -> verify_sliceIntExp'\
	| gofmt -r='verify_patternExp -> verify_patternIntExp'\
	| gofmt -r='make_vectorExp -> make_vectorIntExp'\
	| gofmt -r='TestInsertVectorExp -> TestIntInsertVectorExp'\
	| gofmt -r='TestDoExp -> TestIntDoExp'\
	| gofmt -r='TestIterExp -> TestIntIterExp'\
	| gofmt -r='TestVectorData -> TestIntVectorData'\
	> intvector_test.go\
	
	< vector_test.go cat\
	| gofmt -r='Vector -> StringVector'\
	| gofmt -r='zero -> strzero'\
	| gofmt -r='int2Value -> int2StrValue'\
	| gofmt -r='intf2Value -> intf2StrValue'\
	| gofmt -r='elem2Value -> elem2StrValue'\
	| gofmt -r='TestZeroLenExp -> TestStrZeroLenExp'\
	| gofmt -r='TestResizeExp -> TestStrResizeExp'\
	| gofmt -r='TestResize2Exp -> TestStrResize2Exp'\
	| gofmt -r='checkZeroExp -> checkStrZeroExp'\
	| gofmt -r='TestTrailingElementsExp -> TestStrTrailingElementsExp'\
	| gofmt -r='TestAccessExp -> TestStrAccessExp'\
	| gofmt -r='TestInsertDeleteClearExp -> TestStrInsertDeleteClearExp'\
	| gofmt -r='verify_sliceExp -> verify_sliceStrExp'\
	| gofmt -r='verify_patternExp -> verify_patternStrExp'\
	| gofmt -r='make_vectorExp -> make_vectorStrExp'\
	| gofmt -r='TestInsertVectorExp -> TestStrInsertVectorExp'\
	| gofmt -r='TestDoExp -> TestStrDoExp'\
	| gofmt -r='TestIterExp -> TestStrIterExp'\
	| gofmt -r='TestVectorData -> TestStrVectorData'\
	> stringvector_test.go

include ../../../Make.pkg
