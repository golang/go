// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Test handling of bitfields.

/*
typedef struct {
	unsigned long long f8  : 8;
	unsigned long long f16 : 16;
	unsigned long long f24 : 24;
	unsigned long long f32 : 32;
	unsigned long long f40 : 40;
	unsigned long long f48 : 48;
	unsigned long long f56 : 56;
	unsigned long long f64 : 64;
} issue22958Type;
*/
import "C"

// Nothing to run, just make sure this compiles.
var Vissue22958 C.issue22958Type
