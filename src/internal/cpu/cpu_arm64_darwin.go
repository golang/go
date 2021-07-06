// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 && darwin && !ios
// +build arm64,darwin,!ios

package cpu

func osInit() {
	ARM64.HasATOMICS = sysctlEnabled([]byte("hw.optional.armv8_1_atomics\x00"))
	ARM64.HasCRC32 = sysctlEnabled([]byte("hw.optional.armv8_crc32\x00"))

	// There are no hw.optional sysctl values for the below features on Mac OS 11.0
	// to detect their supported state dynamically. Assume the CPU features that
	// Apple Silicon M1 supports to be available as a minimal set of features
	// to all Go programs running on darwin/arm64.
	ARM64.HasAES = true
	ARM64.HasPMULL = true
	ARM64.HasSHA1 = true
	ARM64.HasSHA2 = true
}

//go:noescape
func getsysctlbyname(name []byte) (int32, int32)

func sysctlEnabled(name []byte) bool {
	ret, value := getsysctlbyname(name)
	if ret < 0 {
		return false
	}
	return value > 0
}
