// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Derived and simplified from (TestSkimageLabToRGBOutOfGamut):
//   https://github.com/go-gfx/gfx/blob/main/color/skimage_test.go#L113C6-L113C35

import (
	"fmt"
	"math"
)

type Lab struct {
	L, A, B float64
}

func QuantizeUnitToByte(v float64) uint8 {
	if v <= 0 {
		return 0
	}

	if v >= 1 {
		return 255
	}

	return uint8(math.RoundToEven(v * 255))
}

func main() {
	labs := []Lab{
		{0.7222675830801158, -0.9031940815103348, 1.301887514238097},
		{0.5782818712019121, 0.4553904020428021, -0.6231537217892238},
		{1.6473994133908043, 0.699861704410822, 0.3917817900198793},
		{-4.466488923617846, 0.3197751169737742, 0.8014614071926937},
		{0.33561687336240814, 1.0179747852290575, -0.23495942874439518},
	}
	want := [][3]uint8{{184, 0, 255}, {147, 116, 0}, {255, 178, 100}, {0, 82, 204}, {86, 255, 0}}
	for i, l := range labs {
		r, g, b := l.L, l.A, l.B
		got := [3]uint8{QuantizeUnitToByte(r), QuantizeUnitToByte(g), QuantizeUnitToByte(b)}
		if got != want[i] {
			panic(fmt.Sprintf("issue81000: %+v = %v, want %v\n", l, got, want[i]))
		}
	}
}
