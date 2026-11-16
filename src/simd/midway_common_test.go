// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package simd

import (
	"fmt"
	"strings"
	"testing"
)

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
		// A size requests that size.
		{128, "128", 128, false},
		{256, "128", 128, false},
		{256, "256", 256, false},
		{512, "128", 128, false},
		{512, "256", 256, false},
		{512, "512", 512, false},
		// Without SIMD hardware, always emulate.
		{0, "", 128, true},
		{0, "1", 128, true},
		{0, "0", 128, true},
		{0, "256", 128, true},
	} {
		size, emulated := configure(test.hwMax, test.value)
		if size != test.wantSize || emulated != test.wantEmulated {
			t.Errorf("configure(%d, %q) = (%d, %t), want (%d, %t)",
				test.hwMax, test.value, size, emulated, test.wantSize, test.wantEmulated)
		}
	}
}

func TestConfigureInvalid(t *testing.T) {
	for _, test := range []struct {
		hwMax int
		value string
		want  string
	}{
		{512, "17", "is not supported"},
		{512, "64", "is not supported"},
		{512, "127", "is not supported"},
		{512, "129", "is not supported"},
		{512, "200", "is not supported"},
		{512, "1024", "is not supported"},
		{512, "-1", "is not supported"},
		{512, "abc", "is not supported"},
		{512, "0128", "is not supported"},
		// The "+" prefix used to override a limit on the vector size
		// that no longer exists.
		{512, "+", "is not supported"},
		{512, "+256", "is not supported"},
		{512, "+17", "is not supported"},
		// "@" used to select a variant, which no longer exist.
		{128, "@nclm", "is not supported"},
		// Sizes the hardware doesn't support.
		{128, "256", "is larger than the SIMD length (128)"},
		{128, "512", "is larger than the SIMD length (128)"},
		{256, "512", "is larger than the SIMD length (256)"},
	} {
		t.Run(fmt.Sprintf("%d/%s", test.hwMax, test.value), func(t *testing.T) {
			defer func() {
				got := recover()
				if got == nil {
					t.Fatalf("configure(%d, %q) did not panic", test.hwMax, test.value)
				}
				if message := fmt.Sprint(got); !strings.Contains(message, test.want) {
					t.Fatalf("configure(%d, %q) panicked with %q, want substring %q", test.hwMax, test.value, message, test.want)
				}
			}()
			configure(test.hwMax, test.value)
		})
	}
}
