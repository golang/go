// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x

package cpu

const CacheLineSize = 256

// the following cpu feature detection functions are defined in cpu_s390x.s
func hasKM() bool
func hasKMC() bool
func hasKMCTR() bool
func hasKMA() bool
func hasKIMD() bool

func init() {
	S390X.HasKM = hasKM()
	S390X.HasKMC = hasKMC()
	S390X.HasKMCTR = hasKMCTR()
	S390X.HasKMA = hasKMA()
	S390X.HasKIMD = hasKIMD()
}
