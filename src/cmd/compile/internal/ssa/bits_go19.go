// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

package ssa

import "math/bits"

func TrailingZeros64(x uint64) int {
	return bits.TrailingZeros64(x)
}
