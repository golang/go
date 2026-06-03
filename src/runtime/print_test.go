// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math"
	"runtime"
	"testing"
)

func FuzzPrintFloat64(f *testing.F) {
	f.Add(math.SmallestNonzeroFloat64)
	f.Add(math.MaxFloat64)
	f.Add(-1.7976931348623157e+308) // requires 24 digits

	f.Fuzz(func(t *testing.T, v float64) {
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Float64Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Float64Bytes)
		}
	})
}

func FuzzPrintFloat32(f *testing.F) {
	f.Add(float32(math.SmallestNonzeroFloat32))
	f.Add(float32(math.MaxFloat32))
	f.Add(float32(-1.06338233e+37)) // requires 15 digits

	f.Fuzz(func(t *testing.T, v float32) {
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Float32Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Float32Bytes)
		}
	})
}

func FuzzPrintComplex128(f *testing.F) {
	f.Add(math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64)
	f.Add(math.MaxFloat64, math.MaxFloat64)
	f.Add(-1.7976931348623157e+308, -1.7976931348623157e+308) // requires 51 digits

	f.Fuzz(func(t *testing.T, r, i float64) {
		v := complex(r, i)
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Complex128Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Complex128Bytes)
		}
	})
}

func FuzzPrintComplex64(f *testing.F) {
	f.Add(float32(math.SmallestNonzeroFloat32), float32(math.SmallestNonzeroFloat32))
	f.Add(float32(math.MaxFloat32), float32(math.MaxFloat32))
	f.Add(float32(-1.06338233e+37), float32(-1.06338233e+37)) // requires 33 digits

	f.Fuzz(func(t *testing.T, r, i float32) {
		v := complex(r, i)
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Complex64Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Complex64Bytes)
		}
	})
}
