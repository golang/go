// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// golang.org/issue/12686.
// interesting because it's a non-constant but ideal value
// and we used to incorrectly attach a constant Val to the Node.

package p

func f(i uint) uint {
	x := []uint{1 << i}
	return x[0]
}
