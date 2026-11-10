// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lib

import "simd"

func F() int {
	return 42
}

func unexported() simd.Uint8s {
	return simd.BroadcastUint8s(1)
}
