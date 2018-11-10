// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#cgo CFLAGS: -Werror
const struct { int a; } *issue4857() { return (void *)0; }
*/
import "C"

func test4857() {
	_ = C.issue4857()
}
