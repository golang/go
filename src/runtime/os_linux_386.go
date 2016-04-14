// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	_AT_RANDOM  = 25
	_AT_SYSINFO = 32
)

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_RANDOM:
		startupRandomData = (*[16]byte)(unsafe.Pointer(val))[:]
	}
}
