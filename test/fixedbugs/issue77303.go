// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 77303: compiler crash on array of zero-size ASPECIAL elements.

package p

type zeroSizeSpecial struct {
	_ [0]float64
}

var x [3]zeroSizeSpecial

func f() bool {
	return x == x
}
