// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
typedef struct
{
        struct
        {
            int x;
        } y[16];
} z;
*/
import "C"

func test6472() {
	// nothing to run, just make sure this compiles
	s := new(C.z)
	println(s.y[0].x)
}
