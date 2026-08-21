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

func TestConfigurePlainPlus(t *testing.T) {
	for _, test := range []struct {
		name           string
		actualMax      int
		allFeatureSize int
		wantHWClmul    bool
	}{
		{"missing feature", 256, 128, false},
		{"all features", 256, 256, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			max, emulated, hwClmul := configure(test.actualMax, test.allFeatureSize, "+")
			if max != test.actualMax || emulated || hwClmul != test.wantHWClmul {
				t.Errorf("configure(%d, %d, +) = (%d, %t, %t), want (%d, false, %t)",
					test.actualMax, test.allFeatureSize, max, emulated, hwClmul,
					test.actualMax, test.wantHWClmul)
			}

			maxWithSize, emulatedWithSize, hwClmulWithSize := configure(test.actualMax, test.allFeatureSize, "+256")
			if max != maxWithSize || emulated != emulatedWithSize || hwClmul != hwClmulWithSize {
				t.Errorf("plain + result (%d, %t, %t) differs from +256 result (%d, %t, %t)",
					max, emulated, hwClmul, maxWithSize, emulatedWithSize, hwClmulWithSize)
			}
		})
	}
}

func TestConfigureDefault(t *testing.T) {
	max, emulated, hwClmul := configure(256, 256, "")
	if max != 256 || emulated || !hwClmul {
		t.Errorf("configure(256, 256, empty) = (%d, %t, %t), want (256, false, true)",
			max, emulated, hwClmul)
	}
}

func TestConfigureOne(t *testing.T) {
	for _, test := range []struct {
		actualMax      int
		allFeatureSize int
	}{
		{128, 0},
		{128, 128},
		{256, 128},
		{256, 256},
		{512, 256},
		{512, 512},
	} {
		gotMax, gotEmulated, gotHWClmul := configure(test.actualMax, test.allFeatureSize, "1")
		wantMax, wantEmulated, wantHWClmul := configure(test.actualMax, test.allFeatureSize, "+")
		if gotMax != wantMax || gotEmulated != wantEmulated || gotHWClmul != wantHWClmul {
			t.Errorf("configure(%d, %d, 1) = (%d, %t, %t), want plain + result (%d, %t, %t)",
				test.actualMax, test.allFeatureSize, gotMax, gotEmulated, gotHWClmul,
				wantMax, wantEmulated, wantHWClmul)
		}
	}
}

func TestConfigureInvalidSize(t *testing.T) {
	for _, test := range []struct {
		value string
		want  string
	}{
		{"17", "not a supported vector size"},
		{"64", "not a supported vector size"},
		{"127", "not a supported vector size"},
		{"129", "not a supported vector size"},
		{"200", "not a supported vector size"},
		{"+17", "not a supported vector size"},
		{"-1", "is negative"},
		{"abc", "could not parse"},
	} {
		t.Run(test.value, func(t *testing.T) {
			defer func() {
				got := recover()
				if got == nil {
					t.Fatalf("configure(512, 512, %q) did not panic", test.value)
				}
				if message := fmt.Sprint(got); !strings.Contains(message, test.want) {
					t.Fatalf("configure(512, 512, %q) panicked with %q, want substring %q", test.value, message, test.want)
				}
			}()
			configure(512, 512, test.value)
		})
	}
}
