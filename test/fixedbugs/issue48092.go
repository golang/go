// compile -B

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure that we can at least compile this code
// successfully with -B. We can't ever produce the right
// answer at runtime with -B, as the access must panic.

package p

type A [0]byte

func (a *A) Get(i int) byte {
	return a[i]
}
