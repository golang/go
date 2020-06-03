// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/cpu"

const (
	// bit masks taken from bits/hwcap.h
	_HWCAP_S390_VX = 2048 // vector facility
)

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_HWCAP: // CPU capability bit flags
		cpu.S390X.HasVX = val&_HWCAP_S390_VX != 0
	}
}

func osArchInit() {}
