// compile

//go:build amd64 && goexperiment.simd

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for ICE on a package-scope var initializer that calls a
// simd function. The midway deep copier built the emulation copy with
// the original selector name node, so the re-check after the rewrite
// resolved the original simd.BroadcastUint8s to the bridge object
// while its cached type still held the simd signature. See issue 80689.

package p

import "simd"

var b = simd.BroadcastUint8s(0x20)

func f() simd.Uint8s {
	return b.Or(b)
}
