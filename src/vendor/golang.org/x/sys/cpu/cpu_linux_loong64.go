// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

// HWCAP bits. These are exposed by the Linux kernel.
const (
	hwcap_LOONGARCH_LSX  = 1 << 4
	hwcap_LOONGARCH_LASX = 1 << 5
)

func doinit() {
	// TODO: Features that require kernel support like LSX and LASX can
	// be detected here once needed in std library or by the compiler.
	Loong64.HasLSX = hwcIsSet(hwCap, hwcap_LOONGARCH_LSX)
	Loong64.HasLASX = hwcIsSet(hwCap, hwcap_LOONGARCH_LASX)
}

func hwcIsSet(hwc uint, val uint) bool {
	return hwc&val != 0
}
