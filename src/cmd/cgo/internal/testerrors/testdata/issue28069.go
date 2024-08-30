// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the error message for an unrepresentable typedef in a
// union appears on the right line. This test is only run if the size
// of long double is larger than 64.

package main

/*
typedef long double             Float128;

typedef struct SV {
    union {
        Float128         float128;
    } value;
} SV;
*/
import "C"

type ts struct {
	tv *C.SV // ERROR HERE
}

func main() {}
