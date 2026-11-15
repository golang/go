// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package config

import (
	"internal/cpu"
	"slices"
	"testing"
)

func TestArchLevelsDefaultUsesV82(t *testing.T) {
	amd64, arm64 := ArchLevels("")
	if amd64 != "v3" || arm64 != "v8.2" {
		t.Fatalf("ArchLevels(%q) = (%q, %q), want (v3, v8.2)", "", amd64, arm64)
	}
	amd64, arm64 = ArchLevels("deterministic")
	if amd64 != "v3" || arm64 != "v8.2" {
		t.Fatalf("ArchLevels(%q) = (%q, %q), want (v3, v8.2)", "deterministic", amd64, arm64)
	}
	amd64, arm64 = ArchLevels("baseline")
	if amd64 != "v1" || arm64 != "v8.0" {
		t.Fatalf("ArchLevels(%q) = (%q, %q), want (v1, v8.0)", "baseline", amd64, arm64)
	}
}

func x86Features() map[string]*bool {
	return map[string]*bool{
		"sse3":     &cpu.X86.HasSSE3,
		"ssse3":    &cpu.X86.HasSSSE3,
		"cx16":     &cpu.X86.HasCX16,
		"sse4.1":   &cpu.X86.HasSSE41,
		"sse4.2":   &cpu.X86.HasSSE42,
		"popcnt":   &cpu.X86.HasPOPCNT,
		"lahf":     &cpu.X86.HasLAHF,
		"fma":      &cpu.X86.HasFMA,
		"movbe":    &cpu.X86.HasMOVBE,
		"osxsave":  &cpu.X86.HasOSXSAVE,
		"avx":      &cpu.X86.HasAVX,
		"f16c":     &cpu.X86.HasF16C,
		"lzcnt":    &cpu.X86.HasLZCNT,
		"bmi1":     &cpu.X86.HasBMI1,
		"avx2":     &cpu.X86.HasAVX2,
		"bmi2":     &cpu.X86.HasBMI2,
		"avx512f":  &cpu.X86.HasAVX512F,
		"avx512dq": &cpu.X86.HasAVX512DQ,
		"avx512cd": &cpu.X86.HasAVX512CD,
		"avx512bw": &cpu.X86.HasAVX512BW,
		"avx512vl": &cpu.X86.HasAVX512VL,
	}
}

func without(set []string, drop string) []string {
	return slices.DeleteFunc(slices.Clone(set), func(s string) bool { return s == drop })
}

func TestNativeAMD64PicksHostLevel(t *testing.T) {
	saved := cpu.X86
	t.Cleanup(func() { cpu.X86 = saved })
	v2 := []string{"sse3", "ssse3", "cx16", "sse4.1", "sse4.2", "popcnt", "lahf"}
	v3 := append(slices.Clone(v2), "fma", "movbe", "osxsave", "avx", "f16c", "lzcnt", "bmi1", "avx2", "bmi2")
	v4 := append(slices.Clone(v3), "avx512f", "avx512dq", "avx512cd", "avx512bw", "avx512vl")
	sandyBridge := append(slices.Clone(v2), "osxsave", "avx")
	for _, tc := range []struct {
		name string
		host []string
		want string
	}{
		{"none", nil, "v1"},
		{"v2-without-cx16", without(v2, "cx16"), "v1"},
		{"sandy-bridge", sandyBridge, "v2"},
		{"v3-without-movbe", without(v3, "movbe"), "v2"},
		{"v3", v3, "v3"},
		{"v4-without-f16c", without(v4, "f16c"), "v2"},
		{"v4", v4, "v4"},
	} {
		features := x86Features()
		for _, f := range features {
			*f = false
		}
		for _, name := range tc.host {
			*features[name] = true
		}
		if got := nativeAMD64(); got != tc.want {
			t.Errorf("%s: nativeAMD64() = %q, want %q", tc.name, got, tc.want)
		}
	}
}
