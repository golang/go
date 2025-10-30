// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

type Uint128 = uint128

var (
	MulLog10_2       = mulLog10_2
	MulLog2_10       = mulLog2_10
	ParseFloatPrefix = parseFloatPrefix
	Pow10            = pow10
	Umul128          = umul128
	Umul192          = umul192
)

func NewDecimal(i uint64) *decimal {
	d := new(decimal)
	d.Assign(i)
	return d
}

func SetOptimize(b bool) bool {
	old := optimize
	optimize = b
	return old
}
