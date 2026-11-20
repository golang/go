// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import "testing"

func TestSIMDImpliesSIMDStable(t *testing.T) {
	for _, test := range []struct {
		goexp      string
		simd       bool
		simdStable bool
		err        bool
	}{
		{goexp: "", simdStable: true},
		{goexp: "simdstable", simdStable: true},
		{goexp: "simd", simd: true, simdStable: true},
		{goexp: "simd,simdstable", simd: true, simdStable: true},
		{goexp: "simdstable,nosimd", simdStable: true},
		{goexp: "simd,nosimd", simdStable: true},
		{goexp: "nosimd", simdStable: true},
		{goexp: "nosimdstable"},
		{goexp: "nosimdstable,simdstable", simdStable: true},
		{goexp: "nosimdstable,nosimd"},
		{goexp: "none"},
		{goexp: "simd,nosimdstable", err: true},
		{goexp: "nosimdstable,simd", err: true},
		{goexp: "nosimdstable,none,simd", simd: true, simdStable: true},
	} {
		flags, err := ParseGOEXPERIMENT("linux", "amd64", test.goexp)
		if test.err {
			if err == nil {
				t.Errorf("ParseGOEXPERIMENT(%q): want error, got nil", test.goexp)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseGOEXPERIMENT(%q): %v", test.goexp, err)
			continue
		}
		if flags.SIMD != test.simd || flags.SIMDStable != test.simdStable {
			t.Errorf("ParseGOEXPERIMENT(%q): SIMD=%v, SIMDStable=%v; want %v, %v", test.goexp, flags.SIMD, flags.SIMDStable, test.simd, test.simdStable)
		}
	}
}
