// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !arm64,!s390x,!ppc64le arm64,!go1.11 gccgo appengine

package chacha20

const bufSize = blockSize

func (s *Cipher) xorKeyStreamBlocks(dst, src []byte) {
	s.xorKeyStreamBlocksGeneric(dst, src)
}
