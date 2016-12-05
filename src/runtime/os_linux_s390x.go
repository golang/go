// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
)

const (
	// bit masks taken from bits/hwcap.h
	_HWCAP_S390_VX = 2048 // vector facility
)

// facilities is padded to avoid false sharing.
type facilities struct {
	_     [sys.CacheLineSize]byte
	hasVX bool // vector facility
	_     [sys.CacheLineSize]byte
}

// cpu indicates the availability of s390x facilities that can be used in
// Go assembly but are optional on models supported by Go.
var cpu facilities

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_HWCAP: // CPU capability bit flags
		cpu.hasVX = val&_HWCAP_S390_VX != 0
	}
}
