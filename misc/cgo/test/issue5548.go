// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "testing"

/*
extern int issue5548_in_c(void);
*/
import "C"

//export issue5548FromC
func issue5548FromC(s string, i int) int {
	if len(s) == 4 && s == "test" && i == 42 {
		return 1
	}
	return 0
}

func test5548(t *testing.T) {
	if C.issue5548_in_c() == 0 {
		t.Fail()
	}
}
