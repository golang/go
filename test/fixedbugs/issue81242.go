// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const W = 32 << (^uintptr(0) >> 63) // 32 or 64

type T struct {
	a [1<<(W-30) - 1]byte
}

func f(t *T) {
	*t = T{}
}
