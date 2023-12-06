// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !gc || purego || !s390x

package sha3

// newShake128Asm returns an assembly implementation of SHAKE-128 if available,
// otherwise it returns nil.
func newShake128Asm() ShakeHash {
	return nil
}

// newShake256Asm returns an assembly implementation of SHAKE-256 if available,
// otherwise it returns nil.
func newShake256Asm() ShakeHash {
	return nil
}
