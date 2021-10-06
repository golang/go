// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

func Int31nForTest(r *Rand, n int32) int32 {
	return r.int31n(n)
}

func GetNormalDistributionParameters() (float64, [128]uint32, [128]float32, [128]float32) {
	return rn, kn, wn, fn
}

func GetExponentialDistributionParameters() (float64, [256]uint32, [256]float32, [256]float32) {
	return re, ke, we, fe
}
