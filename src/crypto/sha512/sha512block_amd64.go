// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

package sha512

import "internal/cpu"

//go:noescape
func blockAVX2(dig *digest, p []byte)

//go:noescape
func blockAMD64(dig *digest, p []byte)

var useAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasBMI1 && cpu.X86.HasBMI2

func block(dig *digest, p []byte) {
	if useAVX2 {
		blockAVX2(dig, p)
	} else {
		blockAMD64(dig, p)
	}
}
