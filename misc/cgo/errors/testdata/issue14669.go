// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 14669: test that fails when build with CGO_CFLAGS selecting
// optimization.

package p

/*
const int E = 1;

typedef struct s {
	int       c;
} s;
*/
import "C"

func F() {
	_ = C.s{
		c: C.E,
	}
}
