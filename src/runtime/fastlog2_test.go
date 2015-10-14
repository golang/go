// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math"
	"runtime"
	"testing"
)

func TestFastLog2(t *testing.T) {
	// Compute the euclidean distance between math.Log2 and the FastLog2
	// implementation over the range of interest for heap sampling.
	const randomBitCount = 26
	var e float64

	inc := 1
	if testing.Short() {
		// Check 1K total values, down from 64M.
		inc = 1 << 16
	}
	for i := 1; i < 1<<randomBitCount; i += inc {
		l, fl := math.Log2(float64(i)), runtime.Fastlog2(float64(i))
		d := l - fl
		e += d * d
	}
	e = math.Sqrt(e)

	if e > 1.0 {
		t.Fatalf("imprecision on fastlog2 implementation, want <=1.0, got %f", e)
	}
}
