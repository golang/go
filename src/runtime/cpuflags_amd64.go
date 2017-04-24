// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

var useAVXmemmove bool

func init() {
	// Let's remove stepping and reserved fields
	processorVersionInfo := cpuid_eax & 0x0FFF3FF0

	isIntelBridgeFamily := isIntel &&
		(processorVersionInfo == 0x206A0 ||
			processorVersionInfo == 0x206D0 ||
			processorVersionInfo == 0x306A0 ||
			processorVersionInfo == 0x306E0)

	useAVXmemmove = support_avx && !isIntelBridgeFamily
}
