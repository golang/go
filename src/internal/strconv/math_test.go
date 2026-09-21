// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"math"
	"testing"
)

func TestLog10Pow2(t *testing.T) {
	for x := -1600; x <= +1600; x++ {
		iMath := log10Pow2(x)
		fMath := int(math.Floor(float64(x) * math.Ln2 / math.Ln10))
		if iMath != fMath {
			t.Errorf("log10Pow2(%d) = %d, want %d\n", x, iMath, fMath)
		}
	}
}

func TestLog2Pow10(t *testing.T) {
	for x := -500; x <= +500; x++ {
		iMath := log2Pow10(x)
		fMath := int(math.Floor(float64(x) * math.Ln10 / math.Ln2))
		if iMath != fMath {
			t.Errorf("log2Pow10(%d) = %d, want %d\n", x, iMath, fMath)
		}
	}
}
