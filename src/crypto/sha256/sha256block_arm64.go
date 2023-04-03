// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha256

import (
	"internal/cpu"
	"unsafe"
)

var k = _K

//go:noescape
func sha256block(h []uint32, p *byte, n int, k []uint32)

func block(dig *digest, p []byte) {
	if !cpu.ARM64.HasSHA2 {
		blockGeneric(dig, p)
	} else {
		h := dig.h[:]
		sha256block(h, unsafe.SliceData(p), len(p), k)
	}
}

func blockString(dig *digest, s string) {
	if !cpu.ARM64.HasSHA2 {
		blockGeneric(dig, s)
	} else {
		h := dig.h[:]
		sha256block(h, unsafe.StringData(s), len(s), k)
	}
}
