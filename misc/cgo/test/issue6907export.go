// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
extern int CheckIssue6907C(_GoString_);
*/
import "C"

import (
	"testing"
)

const CString = "C string"

//export CheckIssue6907Go
func CheckIssue6907Go(s string) C.int {
	if s == CString {
		return 1
	}
	return 0
}

func test6907Go(t *testing.T) {
	if got := C.CheckIssue6907C(CString); got != 1 {
		t.Errorf("C.CheckIssue6907C() == %d, want %d", got, 1)
	}
}
