// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.simd

package crc32

const usePMULL = false

// castagnoliUpdatePMULL never folds without the simd experiment.
func castagnoliUpdatePMULL(crc uint32, p []byte) uint32 {
	return castagnoliUpdate(crc, p)
}
