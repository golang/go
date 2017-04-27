// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

var useAVXmemmove bool

func init() {
	// Let's remove stepping and reserved fields
	processor := processorVersionInfo & 0x0FFF3FF0

	isIntelBridgeFamily := isIntel &&
		processor == 0x206A0 ||
		processor == 0x206D0 ||
		processor == 0x306A0 ||
		processor == 0x306E0

	useAVXmemmove = support_avx && !isIntelBridgeFamily
}
