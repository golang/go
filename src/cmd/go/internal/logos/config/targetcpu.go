// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import (
	"internal/cpu"
	"slices"
)

func ArchLevels(targetCPU string) (amd64, arm64 string) {
	switch targetCPU {
	case "baseline":
		return "v1", "v8.0"
	case "native":
		return nativeAMD64(), nativeARM64()
	default:
		return "v3", "v8.2"
	}
}

func nativeAMD64() string {
	x := &cpu.X86
	levels := []struct {
		name string
		need []bool
	}{
		{"v2", []bool{x.HasSSE3, x.HasSSSE3, x.HasCX16, x.HasSSE41, x.HasSSE42, x.HasPOPCNT, x.HasLAHF}},
		{"v3", []bool{x.HasFMA, x.HasMOVBE, x.HasOSXSAVE, x.HasAVX, x.HasF16C, x.HasLZCNT, x.HasBMI1, x.HasAVX2, x.HasBMI2}},
		{"v4", []bool{x.HasAVX512F, x.HasAVX512DQ, x.HasAVX512CD, x.HasAVX512BW, x.HasAVX512VL}},
	}
	host := "v1"
	for _, level := range levels {
		if slices.Contains(level.need, false) {
			break
		}
		host = level.name
	}
	return host
}

func nativeARM64() string {
	if cpu.ARM64.HasATOMICS {
		return "v8.1"
	}
	return "v8.0"
}
