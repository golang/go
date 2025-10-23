// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import . "strconv"

type uint128 = Uint128

var (
	baseError        = BaseError
	bitSizeError     = BitSizeError
	mulLog10_2       = MulLog10_2
	mulLog2_10       = MulLog2_10
	parseFloatPrefix = ParseFloatPrefix
	pow10            = Pow10
	umul128          = Umul128
	umul192          = Umul192
)
