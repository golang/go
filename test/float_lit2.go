// run

// Check conversion of constant to float32/float64 near min/max boundaries.

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

const (
	m32bits   = 23  // number of float32 mantissa bits
	e32max    = 127 // max. float32 exponent
	maxExp32  = e32max - m32bits
	maxMant32 = 1<<(m32bits+1) - 1

	maxFloat32_0 = (maxMant32 - 0) << maxExp32
	maxFloat32_1 = (maxMant32 - 1) << maxExp32
	maxFloat32_2 = (maxMant32 - 2) << maxExp32
)

func init() {
	if maxExp32 != 104 {
		panic("incorrect maxExp32")
	}
	if maxMant32 != 16777215 {
		panic("incorrect maxMant32")
	}
	if maxFloat32_0 != 340282346638528859811704183484516925440 {
		panic("incorrect maxFloat32_0")
	}
}

const (
	m64bits   = 52   // number of float64 mantissa bits
	e64max    = 1023 // max. float64 exponent
	maxExp64  = e64max - m64bits
	maxMant64 = 1<<(m64bits+1) - 1

	// These expressions are not permitted due to implementation restrictions.
	// maxFloat64_0 = (maxMant64-0) << maxExp64
	// maxFloat64_1 = (maxMant64-1) << maxExp64
	// maxFloat64_2 = (maxMant64-2) << maxExp64

	// These equivalent values were computed using math/big.
	maxFloat64_0 = 1.7976931348623157e308
	maxFloat64_1 = 1.7976931348623155e308
	maxFloat64_2 = 1.7976931348623153e308
)

func init() {
	if maxExp64 != 971 {
		panic("incorrect maxExp64")
	}
	if maxMant64 != 9007199254740991 {
		panic("incorrect maxMant64")
	}
}

var cvt = []struct {
	val    interface{}
	binary string
}{

	{float32(maxFloat32_0), fmt.Sprintf("%dp+%d", int32(maxMant32-0), maxExp32)},
	{float32(maxFloat32_1), fmt.Sprintf("%dp+%d", int32(maxMant32-1), maxExp32)},
	{float32(maxFloat32_2), fmt.Sprintf("%dp+%d", int32(maxMant32-2), maxExp32)},

	{float64(maxFloat64_0), fmt.Sprintf("%dp+%d", int64(maxMant64-0), maxExp64)},
	{float64(maxFloat64_1), fmt.Sprintf("%dp+%d", int64(maxMant64-1), maxExp64)},
	{float64(maxFloat64_2), fmt.Sprintf("%dp+%d", int64(maxMant64-2), maxExp64)},

	{float32(-maxFloat32_0), fmt.Sprintf("-%dp+%d", int32(maxMant32-0), maxExp32)},
	{float32(-maxFloat32_1), fmt.Sprintf("-%dp+%d", int32(maxMant32-1), maxExp32)},
	{float32(-maxFloat32_2), fmt.Sprintf("-%dp+%d", int32(maxMant32-2), maxExp32)},

	{float64(-maxFloat64_0), fmt.Sprintf("-%dp+%d", int64(maxMant64-0), maxExp64)},
	{float64(-maxFloat64_1), fmt.Sprintf("-%dp+%d", int64(maxMant64-1), maxExp64)},
	{float64(-maxFloat64_2), fmt.Sprintf("-%dp+%d", int64(maxMant64-2), maxExp64)},
}

func main() {
	bug := false
	for i, c := range cvt {
		s := fmt.Sprintf("%b", c.val)
		if s != c.binary {
			if !bug {
				bug = true
				fmt.Println("BUG")
			}
			fmt.Printf("#%d: have %s, want %s\n", i, s, c.binary)
		}
	}
}
