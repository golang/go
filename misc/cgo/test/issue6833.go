// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
extern unsigned long long issue6833Func(unsigned int, unsigned long long);
*/
import "C"

import "testing"

//export GoIssue6833Func
func GoIssue6833Func(aui uint, aui64 uint64) uint64 {
	return aui64 + uint64(aui)
}

func test6833(t *testing.T) {
	ui := 7
	ull := uint64(0x4000300020001000)
	v := uint64(C.issue6833Func(C.uint(ui), C.ulonglong(ull)))
	exp := uint64(ui) + ull
	if v != exp {
		t.Errorf("issue6833Func() returns %x, expected %x", v, exp)
	}
}
