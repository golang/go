// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm,!gccgo,!appengine,!nacl

package poly1305

// poly1305_auth_armv6 is implemented in sum_arm.s
//go:noescape
func poly1305_auth_armv6(out *[16]byte, m *byte, mlen uint32, key *[32]byte)

func sum(out *[16]byte, m []byte, key *[32]byte) {
	var mPtr *byte
	if len(m) > 0 {
		mPtr = &m[0]
	}
	poly1305_auth_armv6(out, mPtr, uint32(len(m)), key)
}
