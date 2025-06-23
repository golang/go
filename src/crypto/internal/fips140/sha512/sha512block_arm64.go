// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha512

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var useSHA512 = cpu.ARM64HasSHA512

func init() {
	impl.Register("sha512", "Armv8.2", &useSHA512)
}

//go:noescape
func blockSHA512(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if useSHA512 {
		blockSHA512(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
