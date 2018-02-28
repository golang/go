// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
long double x = 0;
*/
import "C"

func main() {
	_ = C.x // ERROR HERE
	_ = C.x
}
