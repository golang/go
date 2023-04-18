// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func dgemmSerialNotNot(m, n, k int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, alpha float64) {
	for i := 0; i < m; i++ {
		ctmp := c[i*ldc : i*ldc+n]
		for l, v := range a[i*lda : i*lda+k] {
			tmp := alpha * v
			if tmp != 0 {
				x := b[l*ldb : l*ldb+n]
				// amd64:"INCQ"
				for i, v := range x {
					ctmp[i] += tmp * v
				}
			}
		}
	}
}
