// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// +build ignore

package main

// This file tests that when cgo -godefs sees a struct with a field
// that is an anonymous union, the first field in the union is
// promoted to become a field of the struct.  See issue 6677 for
// background.

/*
typedef struct {
	union {
		long l;
		int c;
	};
} t;
*/
import "C"

// Input for cgo -godefs.

type T C.t
