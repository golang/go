// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64

package sha512

import (
	"internal/cpu"
	"unsafe"
)

//go:noescape
func blockAVX2(dig *digest, p *byte, n int)

//go:noescape
func blockAMD64(dig *digest, p *byte, n int)

var useAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasBMI1 && cpu.X86.HasBMI2

func block(dig *digest, p []byte) {
	if useAVX2 {
		blockAVX2(dig, unsafe.SliceData(p), len(p))
	} else {
		blockAMD64(dig, unsafe.SliceData(p), len(p))
	}
}

func blockString(dig *digest, s string) {
	if useAVX2 {
		blockAVX2(dig, unsafe.StringData(s), len(s))
	} else {
		blockAMD64(dig, unsafe.StringData(s), len(s))
	}
}
