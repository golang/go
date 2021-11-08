// compile -N

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(x float64) bool {
	x += 1
	return (x != 0) == (x != 0)
}
