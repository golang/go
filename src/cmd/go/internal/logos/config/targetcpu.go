// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import "internal/cpu"

// ArchLevels maps a target_cpu value to the GOAMD64 and GOARM64 levels
// the build and the LSP hints runner compile with. "baseline" selects the
// oldest levels, "native" probes the host, and any other value, including
// "deterministic" and the empty default, selects the deterministic levels.
// It is the single owner of the target_cpu selection: both the build path
// and the hints path call it, so their levels cannot drift apart.
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

// nativeAMD64 picks the host GOAMD64 level from CPU feature bits.
func nativeAMD64() string {
	if cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW && cpu.X86.HasAVX512CD && cpu.X86.HasAVX512DQ && cpu.X86.HasAVX512VL {
		return "v4"
	}
	if cpu.X86.HasAVX2 && cpu.X86.HasBMI2 && cpu.X86.HasFMA {
		return "v3"
	}
	return "v1"
}

// nativeARM64 picks the host GOARM64 level from CPU feature bits.
func nativeARM64() string {
	if cpu.ARM64.HasATOMICS {
		return "v8.1"
	}
	return "v8.0"
}
