// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 77604: compiler crash when source and destination
// of copy are the same address.

package p

type T struct {
	a [192]byte
}

func f(x *T) {
	i := any(x)
	y := i.(*T)
	*y = *x
}
