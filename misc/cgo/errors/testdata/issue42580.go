// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 42580: cmd/cgo: shifting identifier position in ast

package cgotest

// typedef int (*intFunc) ();
//
// char* strarg = "";
//
// int func_with_char(char* arg, void* dummy)
// {return 5;}
//
// int* get_arr(char* arg, void* dummy)
// {return NULL;}
import "C"
import "unsafe"

// Test variables
var (
	checkedPointer            = []byte{1}
	doublePointerChecked      = []byte{1}
	singleInnerPointerChecked = []byte{1}
)

// This test checks the positions of variable identifiers.
// Changing the positions of the test variables idents after this point will break the test.

func TestSingleArgumentCast() C.int {
	retcode := C.func_with_char((*C.char)(unsafe.Pointer(&checkedPointer[0])), unsafe.Pointer(C.strarg))
	return retcode
}

func TestSingleArgumentCastRecFuncAsSimpleArg() C.int {
	retcode := C.func_with_char((*C.char)(unsafe.Pointer(C.get_arr((*C.char)(unsafe.Pointer(&singleInnerPointerChecked[0])), unsafe.Pointer(C.strarg)))), nil)
	return retcode
}

func TestSingleArgumentCastRecFunc() C.int {
	retcode := C.func_with_char((*C.char)(unsafe.Pointer(C.get_arr((*C.char)(unsafe.Pointer(&doublePointerChecked[0])), unsafe.Pointer(C.strarg)))), unsafe.Pointer(C.strarg))
	return retcode
}
