// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var M map[float64]string

func f() int {
	switch M[0.1] != "a" {
	case true:
		return 1
	default:
		return 0
	}
}
