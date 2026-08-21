// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package simd

import "testing"

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
