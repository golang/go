// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha256

import "internal/cpu"

//go:noescape
func blockAVX2(dig *digest, p []byte)

//go:noescape
func blockAMD64(dig *digest, p []byte)

//go:noescape
func blockSHA(dig *digest, p []byte)

var useAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasBMI2
var useSHA = cpu.X86.HasSHA && cpu.X86.HasSSE41 && cpu.X86.HasSSSE3

func block(dig *digest, p []byte) {
	if useSHA {
		blockSHA(dig, p)
	} else if useAVX2 {
		blockAVX2(dig, p)
	} else {
		blockAMD64(dig, p)
	}
}
