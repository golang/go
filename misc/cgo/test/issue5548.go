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
		return 12345
	}
	println("got", len(s), i)
	return 9876
}

func test5548(t *testing.T) {
	if x := C.issue5548_in_c(); x != 12345 {
		t.Errorf("issue5548_in_c = %d, want %d", x, 12345)
	}
}
