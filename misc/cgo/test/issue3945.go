// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Test that cgo reserves enough stack space during cgo call.
// See https://golang.org/issue/3945 for details.

// #include <stdio.h>
//
// void say() {
//    printf("%s from C\n", "hello");
// }
//
import "C"

import "testing"

func testPrintf(t *testing.T) {
	C.say()
}
