// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import "testing"

func TestSimdInitProxyNames(t *testing.T) {
	tests := []struct {
		target   string
		vecSize  int
		wantName string
		wantBase string
	}{
		{"N", 0, "@simdinit0@N", "N"},
		{"N", 128, "@simdinit128@N", "N"},
		{"N", 256, "@simdinit256@N", "N"},
		{"N", 512, "@simdinit512@N", "N"},
		{"_", 0, "@simdinit0@_", "_"},
		{"_@0", 0, "@simdinit0@_@0", "_"},
		{"_@12", 128, "@simdinit128@_@12", "_"},
	}

	for _, tc := range tests {
		gotName := SimdInitProxyName(tc.target, tc.vecSize)
		if gotName != tc.wantName {
			t.Errorf("SimdInitProxyName(%q, %d) = %q, want %q", tc.target, tc.vecSize, gotName, tc.wantName)
		}
		if !IsSimdInitProxy(gotName) {
			t.Errorf("IsSimdInitProxy(%q) = false, want true", gotName)
		}
		base, size, ok := ParseSimdInitProxy(gotName)
		if !ok || base != tc.wantBase || size != tc.vecSize {
			t.Errorf("ParseSimdInitProxy(%q) = (%q, %d, %v), want (%q, %d, true)",
				gotName, base, size, ok, tc.wantBase, tc.vecSize)
		}
		if _, _, ok := ParseSimdSuffix(gotName); ok {
			t.Errorf("ParseSimdSuffix(%q) = ok, want false for init proxy", gotName)
		}
	}
}

func TestParseSimdInitProxyInvalid(t *testing.T) {
	invalid := []string{
		"",
		"N",
		"N@simd0",
		"N@simd128",
		"@simdinit",
		"@simdinit@",
		"@simdinit0@",
		"@simdinit@N",
		"@simdinitabc@N",
		"@simdinit128abc@N",
		"@simdinit-128@N",
		"prefix@simdinit0@N",
	}
	for _, s := range invalid {
		if base, size, ok := ParseSimdInitProxy(s); ok {
			t.Errorf("ParseSimdInitProxy(%q) = (%q, %d, true), want ok=false", s, base, size)
		}
	}
}

func TestParseSimdSuffix(t *testing.T) {
	valid := []struct {
		sym      string
		wantBase string
		wantSize int
	}{
		{"x@simd0", "x", 0},
		{"x@simd128", "x", 128},
		{"x@simd256", "x", 256},
		{"x@simd512", "x", 512},
	}
	for _, tc := range valid {
		base, size, ok := ParseSimdSuffix(tc.sym)
		if !ok || base != tc.wantBase || size != tc.wantSize {
			t.Errorf("ParseSimdSuffix(%q) = (%q, %d, %v), want (%q, %d, true)",
				tc.sym, base, size, ok, tc.wantBase, tc.wantSize)
		}
		if IsSimdInitProxy(tc.sym) {
			t.Errorf("IsSimdInitProxy(%q) = true, want false", tc.sym)
		}
	}

	invalid := []string{
		"",
		"x",
		"x@simd",
		"x@simdabc",
		"x@simd128abc",
		"x@simd-128",
		"@simdinit0@x",
		"@simdinit128@x",
	}
	for _, s := range invalid {
		if base, size, ok := ParseSimdSuffix(s); ok {
			t.Errorf("ParseSimdSuffix(%q) = (%q, %d, true), want ok=false", s, base, size)
		}
	}
}
