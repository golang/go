// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.4

package constant

import (
	"math"
	"math/big"
)

func ratToFloat32(x *big.Rat) (float32, bool) {
	// Before 1.4, there's no Rat.Float32.
	// Emulate it, albeit at the cost of
	// imprecision in corner cases.
	x64, exact := x.Float64()
	x32 := float32(x64)
	if math.IsInf(float64(x32), 0) {
		exact = false
	}
	return x32, exact
}
