// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

var vendorStringBytes [12]byte
var maxInputValue uint32
var featureFlags uint32
var processorVersionInfo uint32

var useRepMovs = true

func hasFeature(feature uint32) bool {
	return (featureFlags & feature) != 0
}

func cpuid_low(arg1, arg2 uint32) (eax, ebx, ecx, edx uint32) // implemented in cpuidlow_amd64.s
func xgetbv_low(arg1 uint32) (eax, edx uint32)                // implemented in cpuidlow_amd64.s

func init() {
	const cfOSXSAVE uint32 = 1 << 27
	const cfAVX uint32 = 1 << 28

	leaf0()
	leaf1()

	enabledAVX := false
	// Let's check if OS has set CR4.OSXSAVE[bit 18]
	// to enable XGETBV instruction.
	if hasFeature(cfOSXSAVE) {
		eax, _ := xgetbv_low(0)
		// Let's check that XCR0[2:1] = ‘11b’
		// i.e. XMM state and YMM state are enabled by OS.
		enabledAVX = (eax & 0x6) == 0x6
	}

	isIntelBridgeFamily := (processorVersionInfo == 0x206A0 ||
		processorVersionInfo == 0x206D0 ||
		processorVersionInfo == 0x306A0 ||
		processorVersionInfo == 0x306E0) &&
		isIntel()

	useRepMovs = !(hasFeature(cfAVX) && enabledAVX) || isIntelBridgeFamily
}

func leaf0() {
	eax, ebx, ecx, edx := cpuid_low(0, 0)
	maxInputValue = eax
	int32ToBytes(ebx, vendorStringBytes[0:4])
	int32ToBytes(edx, vendorStringBytes[4:8])
	int32ToBytes(ecx, vendorStringBytes[8:12])
}

func leaf1() {
	if maxInputValue < 1 {
		return
	}
	eax, _, ecx, _ := cpuid_low(1, 0)
	// Let's remove stepping and reserved fields
	processorVersionInfo = eax & 0x0FFF3FF0
	featureFlags = ecx
}

func int32ToBytes(arg uint32, buffer []byte) {
	buffer[3] = byte(arg >> 24)
	buffer[2] = byte(arg >> 16)
	buffer[1] = byte(arg >> 8)
	buffer[0] = byte(arg)
}

func isIntel() bool {
	intelSignature := [12]byte{'G', 'e', 'n', 'u', 'i', 'n', 'e', 'I', 'n', 't', 'e', 'l'}
	return vendorStringBytes == intelSignature
}
