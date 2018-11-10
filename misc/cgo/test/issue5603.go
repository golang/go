// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
const long long issue5603exp = 0x12345678;
long long issue5603foo0() { return issue5603exp; }
long long issue5603foo1(void *p) { return issue5603exp; }
long long issue5603foo2(void *p, void *q) { return issue5603exp; }
long long issue5603foo3(void *p, void *q, void *r) { return issue5603exp; }
long long issue5603foo4(void *p, void *q, void *r, void *s) { return issue5603exp; }
*/
import "C"

import "testing"

func test5603(t *testing.T) {
	var x [5]int64
	exp := int64(C.issue5603exp)
	x[0] = int64(C.issue5603foo0())
	x[1] = int64(C.issue5603foo1(nil))
	x[2] = int64(C.issue5603foo2(nil, nil))
	x[3] = int64(C.issue5603foo3(nil, nil, nil))
	x[4] = int64(C.issue5603foo4(nil, nil, nil, nil))
	for i, v := range x {
		if v != exp {
			t.Errorf("issue5603foo%d() returns %v, expected %v", i, v, exp)
		}
	}
}
