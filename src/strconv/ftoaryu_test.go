// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"math"
	. "strconv"
	"testing"
)

func TestMulByLog2Log10(t *testing.T) {
	for x := -1600; x <= +1600; x++ {
		iMath := MulByLog2Log10(x)
		fMath := int(math.Floor(float64(x) * math.Ln2 / math.Ln10))
		if iMath != fMath {
			t.Errorf("mulByLog2Log10(%d) failed: %d vs %d\n", x, iMath, fMath)
		}
	}
}

func TestMulByLog10Log2(t *testing.T) {
	for x := -500; x <= +500; x++ {
		iMath := MulByLog10Log2(x)
		fMath := int(math.Floor(float64(x) * math.Ln10 / math.Ln2))
		if iMath != fMath {
			t.Errorf("mulByLog10Log2(%d) failed: %d vs %d\n", x, iMath, fMath)
		}
	}
}
