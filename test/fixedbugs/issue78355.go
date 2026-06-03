// errorcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64

// Issue 78355: map element or key type too large should not cause ICE.

package p

type T [1 << 31]byte

func F(m map[int]T) { // ERROR "map element type too large"
	_ = m[0]
}
