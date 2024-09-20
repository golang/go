// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha256

import (
	"crypto/internal/impl"
	"internal/cpu"
)

var useSHA2 = cpu.ARM64.HasSHA2

func init() {
	impl.Register("crypto/sha256", "Armv8.0", &useSHA2)
}

//go:noescape
func blockSHA2(dig *digest, p []byte)

func block(dig *digest, p []byte) {
	if useSHA2 {
		blockSHA2(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
