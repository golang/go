// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
//enum test { foo, bar };
*/
import "C"

func main() {
	var a = C.enum_test(1) // ERROR HERE
	_ = a
}
