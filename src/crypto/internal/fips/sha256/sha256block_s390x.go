// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha256

import (
	"crypto/internal/impl"
	"internal/cpu"
)

var useSHA256 = cpu.S390X.HasSHA256

func init() {
	// CP Assist for Cryptographic Functions (CPACF)
	// https://www.ibm.com/docs/en/zos/3.1.0?topic=icsf-cp-assist-cryptographic-functions-cpacf
	impl.Register("crypto/sha256", "CPACF", &useSHA256)
}

//go:noescape
func blockS390X(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if useSHA256 {
		blockS390X(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
