// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#cgo CFLAGS: -Werror=unused-variable
void funcWithoutAnyParams() {}
*/
import "C"

// Only test whether this can be compiled, unused
// variable (e.g. empty gcc strut) could cause
// warning/error under stricter CFLAGS.
func testEmptyGccStruct() {
	C.funcWithoutAnyParams()
}
