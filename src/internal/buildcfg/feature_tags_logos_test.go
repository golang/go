// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import (
	"slices"
	"testing"
)

func TestGogoarchTagsPreservesRawGO386(t *testing.T) {
	oldArch, old386 := GOARCH, GO386
	t.Cleanup(func() { GOARCH, GO386 = oldArch, old386 })
	GOARCH, GO386 = "386", "invalid"
	defer func() {
		if failure := recover(); failure != nil {
			t.Errorf("raw GO386 must reach the ordinary compiler diagnostic, not panic: %v", failure)
		}
	}()
	if got := gogoarchTags(); !slices.Equal(got, []string{"386.invalid"}) {
		t.Errorf("raw GO386 tags = %q, want [386.invalid]", got)
	}
}

func TestGOGOARCHTagsValidatesCandidateSettings(t *testing.T) {
	for _, tc := range []struct {
		arch, level string
		valid       bool
	}{
		{"386", "invalid", false}, {"386", "sse2", true}, {"386", "softfloat", true},
		{"amd64", "v0", false}, {"amd64", "v1", true}, {"amd64", "v2", true},
		{"amd64", "v3", true}, {"amd64", "v4", true}, {"amd64", "v5", false},
		{"arm", "4", false}, {"arm", "5", true}, {"arm", "6", true},
		{"arm", "7", true}, {"arm", "8", false}, {"arm", "7,softfloat", true},
		{"arm64", "v8.0", true}, {"arm64", "v8.9", true}, {"arm64", "v8.10", false},
		{"arm64", "v9.4", true}, {"arm64", "v9.5", true}, {"arm64", "v9.6", false},
		{"mips", "softfloat", true}, {"mipsle", "hardfloat", true}, {"mips", "invalid", false},
		{"mips64", "hardfloat", true}, {"mips64le", "softfloat", true},
		{"ppc64", "power7", false}, {"ppc64", "power8", true}, {"ppc64", "power9", true},
		{"ppc64", "power10", true}, {"ppc64", "power11", false},
		{"riscv64", "rva20u64", true}, {"riscv64", "rva21u64", false},
		{"riscv64", "rva22u64", true}, {"riscv64", "rva23u64", true}, {"riscv64", "rva24u64", false},
		{"wasm", "", true}, {"wasm", "satconv,signext", true}, {"wasm", "invalid", false},
	} {
		t.Run(tc.arch+"/"+tc.level, func(t *testing.T) {
			_, err := GOGOARCHTags(tc.arch, tc.level)
			if (err == nil) != tc.valid {
				t.Fatalf("candidate %s=%q: error=%v, valid=%v", tc.arch, tc.level, err, tc.valid)
			}
		})
	}
}
