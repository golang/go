// compile

//go:build goexperiment.simd

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "simd"

func f(_ int, _ byte, b []byte) int {
	return simd.BroadcastUint8s(' ').StorePart(b)
}

func g(int, byte, []byte) int {
	return simd.BroadcastUint8s(' ').StorePart(nil)
}
