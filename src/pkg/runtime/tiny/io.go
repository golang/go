// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple CGA screen output

package runtime

import "unsafe"

var crt *[25 * 80]uint16
var pos int

func putc(c int) {
	const (
		port  = 0x3d4
		color = 0x0700 // white on black
	)

	if crt == nil {
		// init on demand in case printf is called before
		// initialization runs.
		var mem uintptr = 0xb8000
		crt = (*[25 * 80]uint16)(unsafe.Pointer(mem))
		pos = 0
		for i := range crt[0:] {
			crt[i] = 0
		}
	}

	switch c {
	case '\n':
		pos += 80 - pos%80
	default:
		crt[pos] = uint16(c&0xff | color)
		pos++
	}

	if pos/80 >= 24 {
		copy(crt[0:], crt[80:])
		pos -= 80
		for i := 0; i < 80; i++ {
			crt[24*80+i] = 0
		}
	}
	crt[pos] = ' ' | color
}

func write(fd int32, b []byte) {
	for _, c := range b {
		putc(int(c))
	}
}
