// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha512

import (
	"crypto/internal/impl"
	"internal/cpu"
)

var useSHA512 = cpu.ARM64.HasSHA512

func init() {
	impl.Register("crypto/sha512", "Armv8.2", &useSHA512)
}

//go:noescape
func blockSHA512(dig *digest, p []byte)

func block(dig *digest, p []byte) {
	if useSHA512 {
		blockSHA512(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
