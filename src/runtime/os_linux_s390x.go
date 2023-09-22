// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/cpu"

const (
	_HWCAP_VX = 1 << 11 // vector facility
)

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_HWCAP:
		cpu.HWCap = uint(val)
	}
}

func osArchInit() {}

func checkS390xCPU() {
	// Check if the present z-system has the hardware capability to carryout
	// floating point operations. Check if hwcap reflects CPU capability for the
	// necessary floating point hardware (HasVX) availability.
	// Starting with Go1.19, z13 is the minimum machine level for running Go on LoZ
	if cpu.HWCap&_HWCAP_VX == 0 {
		print("runtime: This CPU has no floating point hardware, so this program cannot be run. \n")
		exit(1)
	}
}
