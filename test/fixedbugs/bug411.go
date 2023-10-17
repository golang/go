// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2588.  Used to trigger internal compiler error on 8g,
// because the compiler tried to registerize the int64 being
// used as a memory operand of a int64->float64 move.

package p

func f1(a int64) {
	f2(float64(a), float64(a))
}

func f2(a,b float64) {
}

