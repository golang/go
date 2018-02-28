// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 13635: used to output error about C.unsignedchar.
// This test tests all such types.

package pkg

import "C"

func main() {
	var (
		_ C.uchar         = "uc"  // ERROR HERE: C\.uchar
		_ C.schar         = "sc"  // ERROR HERE: C\.schar
		_ C.ushort        = "us"  // ERROR HERE: C\.ushort
		_ C.uint          = "ui"  // ERROR HERE: C\.uint
		_ C.ulong         = "ul"  // ERROR HERE: C\.ulong
		_ C.longlong      = "ll"  // ERROR HERE: C\.longlong
		_ C.ulonglong     = "ull" // ERROR HERE: C\.ulonglong
		_ C.complexfloat  = "cf"  // ERROR HERE: C\.complexfloat
		_ C.complexdouble = "cd"  // ERROR HERE: C\.complexdouble
	)
}
