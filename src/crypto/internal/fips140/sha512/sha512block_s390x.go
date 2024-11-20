// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha512

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var useSHA512 = cpu.S390XHasSHA512

func init() {
	// CP Assist for Cryptographic Functions (CPACF)
	// https://www.ibm.com/docs/en/zos/3.1.0?topic=icsf-cp-assist-cryptographic-functions-cpacf
	impl.Register("sha512", "CPACF", &useSHA512)
}

//go:noescape
func blockS390X(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if useSHA512 {
		blockS390X(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
