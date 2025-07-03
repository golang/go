// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 && darwin && !ios

package cpu

import _ "unsafe" // for linkname

func osInit() {
	// macOS 12 moved these to the hw.optional.arm tree, but as of Go 1.24 we
	// still support macOS 11. See [Determine Encryption Capabilities].
	//
	// [Determine Encryption Capabilities]: https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics#3918855
	ARM64.HasATOMICS = sysctlEnabled([]byte("hw.optional.armv8_1_atomics\x00"))
	ARM64.HasCRC32 = sysctlEnabled([]byte("hw.optional.armv8_crc32\x00"))
	ARM64.HasSHA512 = sysctlEnabled([]byte("hw.optional.armv8_2_sha512\x00"))
	ARM64.HasSHA3 = sysctlEnabled([]byte("hw.optional.armv8_2_sha3\x00"))

	ARM64.HasDIT = sysctlEnabled([]byte("hw.optional.arm.FEAT_DIT\x00"))

	// There are no hw.optional sysctl values for the below features on macOS 11
	// to detect their supported state dynamically (although they are available
	// in the hw.optional.arm tree on macOS 12). Assume the CPU features that
	// Apple Silicon M1 supports to be available on all future iterations.
	ARM64.HasAES = true
	ARM64.HasPMULL = true
	ARM64.HasSHA1 = true
	ARM64.HasSHA2 = true
}

//go:noescape
func getsysctlbyname(name []byte) (int32, int32)

// sysctlEnabled should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/gopkg
//   - github.com/songzhibin97/gkit
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname sysctlEnabled
func sysctlEnabled(name []byte) bool {
	ret, value := getsysctlbyname(name)
	if ret < 0 {
		return false
	}
	return value > 0
}
