// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 13635: used to output error about C.unsignedchar.
// This test tests all such types.

package pkg

import "C"

func main() {
	var (
		_ C.uchar         = "uc"  // ERROR HERE
		_ C.schar         = "sc"  // ERROR HERE
		_ C.ushort        = "us"  // ERROR HERE
		_ C.uint          = "ui"  // ERROR HERE
		_ C.ulong         = "ul"  // ERROR HERE
		_ C.longlong      = "ll"  // ERROR HERE
		_ C.ulonglong     = "ull" // ERROR HERE
		_ C.complexfloat  = "cf"  // ERROR HERE
		_ C.complexdouble = "cd"  // ERROR HERE
	)
}
