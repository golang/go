// Copyright (c) 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !purego

package field

func feSquareN(v, a *Element, n int) {
	feSquare(v, a)
	for range n - 1 {
		feSquare(v, v)
	}
}
