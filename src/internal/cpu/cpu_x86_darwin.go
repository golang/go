// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (386 || amd64) && darwin && !ios

package cpu

func osInit() {
	if isRosetta() && darwinKernelVersionCheck(24, 0, 0) {
		// Apparently, on macOS 15 (Darwin kernel version 24) or newer,
		// Rosetta 2 supports AVX1 and 2. However, neither CPUID nor
		// sysctl says it has AVX. Detect this situation here and report
		// AVX1 and 2 as supported.
		// TODO: check if any other feature is actually supported.
		X86.HasAVX = true
		X86.HasAVX2 = true
	}
}

func isRosetta() bool {
	return sysctlEnabled([]byte("sysctl.proc_translated\x00"))
}
