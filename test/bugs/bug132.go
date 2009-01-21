// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	x, x int  // this should be a compile-time error
}

/*
Accessing obj.x for obj of type T will lead to an error so this cannot
be used in a program, but I would argue that this should be a compile-
tume error at the declaration point.
*/
