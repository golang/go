// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd

// sgemmKernelSVE is the hand-written assembly counterpart of the inner loop
// of sgemmSIMD (see sgemm_sve_arm64_test.go).
//
//	cioff is c[i][off:off+vl], ai is a[i], boff is b[0][off:off+vl]
//
//go:noescape
func sgemmKernelSVE(cioff, ai, boff *float32, k, vlx4 int64)
