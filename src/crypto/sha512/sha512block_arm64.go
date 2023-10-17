// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha512

import "internal/cpu"

func block(dig *digest, p []byte) {
	if cpu.ARM64.HasSHA512 {
		blockAsm(dig, p)
		return
	}
	blockGeneric(dig, p)
}

//go:noescape
func blockAsm(dig *digest, p []byte)
