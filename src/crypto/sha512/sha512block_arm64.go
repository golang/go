// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64
// +build arm64

package sha512

//go:noescape
func blockNEON(dig *digest, p []byte)

func block(dig *digest, p []byte) {
	// FIXME using cpu.ARM64.HasSHA512 to protect the devices without SHA512 instructions
	blockNEON(dig, p)
}
