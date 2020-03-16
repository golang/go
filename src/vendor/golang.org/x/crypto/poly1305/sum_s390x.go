// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,go1.11,!gccgo,!appengine

package poly1305

import (
	"golang.org/x/sys/cpu"
)

// poly1305vx is an assembly implementation of Poly1305 that uses vector
// instructions. It must only be called if the vector facility (vx) is
// available.
//go:noescape
func poly1305vx(out *[16]byte, m *byte, mlen uint64, key *[32]byte)

// poly1305vmsl is an assembly implementation of Poly1305 that uses vector
// instructions, including VMSL. It must only be called if the vector facility (vx) is
// available and if VMSL is supported.
//go:noescape
func poly1305vmsl(out *[16]byte, m *byte, mlen uint64, key *[32]byte)

func sum(out *[16]byte, m []byte, key *[32]byte) {
	if cpu.S390X.HasVX {
		var mPtr *byte
		if len(m) > 0 {
			mPtr = &m[0]
		}
		if cpu.S390X.HasVXE && len(m) > 256 {
			poly1305vmsl(out, mPtr, uint64(len(m)), key)
		} else {
			poly1305vx(out, mPtr, uint64(len(m)), key)
		}
	} else {
		sumGeneric(out, m, key)
	}
}
