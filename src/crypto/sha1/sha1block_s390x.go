// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha1

import (
	"crypto/internal/impl"
	"internal/cpu"
)

var useSHA1 = cpu.S390X.HasSHA1

func init() {
	// CP Assist for Cryptographic Functions (CPACF)
	// https://www.ibm.com/docs/en/zos/3.1.0?topic=icsf-cp-assist-cryptographic-functions-cpacf
	impl.Register("sha1", "CPACF", &useSHA1)
}

//go:noescape
func blockS390X(dig *digest, p []byte)

func block(dig *digest, p []byte) {
	if useSHA1 {
		blockS390X(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
