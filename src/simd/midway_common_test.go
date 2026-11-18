// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package simd

import "testing"

func TestConfigure(t *testing.T) {
	for _, test := range []struct {
		hwMax        int
		value        string
		wantSize     int
		wantEmulated bool
	}{
		// By default, use the largest vectors the hardware supports.
		{128, "", 128, false},
		{256, "", 256, false},
		{512, "", 512, false},
		// 1 is the same as the default.
		{128, "1", 128, false},
		{256, "1", 256, false},
		{512, "1", 512, false},
		// 0 requests emulation.
		{128, "0", 128, true},
		{512, "0", 128, true},
		// A size requests at most that size.
		{128, "128", 128, false},
		{256, "128", 128, false},
		{256, "256", 256, false},
		{512, "128", 128, false},
		{512, "256", 256, false},
		{512, "512", 512, false},
		// A size larger than the hardware supports uses the largest
		// size it does support.
		{128, "256", 128, false},
		{128, "512", 128, false},
		{256, "512", 256, false},
		// Unrecognized values are the same as the default.
		{512, "17", 512, false},
		{512, "64", 512, false},
		{512, "127", 512, false},
		{512, "129", 512, false},
		{512, "200", 512, false},
		{512, "1024", 512, false},
		{512, "-1", 512, false},
		{512, "abc", 512, false},
		{512, "0128", 512, false},
		// The "+" prefix used to override a limit on the vector size
		// that no longer exists.
		{512, "+", 512, false},
		{512, "+256", 512, false},
		{512, "+17", 512, false},
		// "@" used to select a variant, which no longer exist.
		{128, "@nclm", 128, false},
		// Without SIMD hardware, always emulate.
		{0, "", 128, true},
		{0, "1", 128, true},
		{0, "0", 128, true},
		{0, "256", 128, true},
		{0, "abc", 128, true},
	} {
		size, emulated := configure(test.hwMax, test.value)
		if size != test.wantSize || emulated != test.wantEmulated {
			t.Errorf("configure(%d, %q) = (%d, %t), want (%d, %t)",
				test.hwMax, test.value, size, emulated, test.wantSize, test.wantEmulated)
		}
	}
}
