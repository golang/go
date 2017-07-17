// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

//void callMulti(void);
import "C"

import "testing"

//export multi
func multi() (*C.char, C.int) {
	return C.CString("multi"), 0
}

func test20910(t *testing.T) {
	C.callMulti()
}
