// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
typedef void callback(char*);
extern void CallGoBigStack1(char*);
extern void bigStack(callback*);
*/
import "C"

func init() {
	register("BigStack", BigStack)
}

func BigStack() {
	// Create a large thread stack and call back into Go to test
	// if Go correctly determines the stack bounds.
	C.bigStack((*C.callback)(C.CallGoBigStack1))
}

//export goBigStack1
func goBigStack1(x *C.char) {
	println("OK")
}
