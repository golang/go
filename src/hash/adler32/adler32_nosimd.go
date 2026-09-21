// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !((amd64 || arm64) && goexperiment.simd)

package adler32

const (
	haveSIMD = false
	minSIMD  = 0
)

func updateSIMD(d digest, p []byte) digest {
	panic("unreachable")
}
