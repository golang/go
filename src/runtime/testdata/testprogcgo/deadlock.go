// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
char *geterror() {
	return "cgo error";
}
*/
import "C"
import (
	"fmt"
)

func init() {
	register("CgoPanicDeadlock", CgoPanicDeadlock)
}

type cgoError struct{}

func (cgoError) Error() string {
	fmt.Print("") // necessary to trigger the deadlock
	return C.GoString(C.geterror())
}

func CgoPanicDeadlock() {
	panic(cgoError{})
}
