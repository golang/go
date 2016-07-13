// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package route

import "unsafe"

var (
	nativeEndian binaryByteOrder
	kernelAlign  int
	parseFns     map[int]parseFn
)

func init() {
	i := uint32(1)
	b := (*[4]byte)(unsafe.Pointer(&i))
	if b[0] == 1 {
		nativeEndian = littleEndian
	} else {
		nativeEndian = bigEndian
	}
	kernelAlign, parseFns = probeRoutingStack()
}

func roundup(l int) int {
	if l == 0 {
		return kernelAlign
	}
	return (l + kernelAlign - 1) & ^(kernelAlign - 1)
}

type parseFn func(RIBType, []byte) (Message, error)

type wireFormat struct {
	extOff  int // offset of header extension
	bodyOff int // offset of message body
}
