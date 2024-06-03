// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !gc || purego || !s390x

package sha3

func newShake128() *state {
	return newShake128Generic()
}

func newShake256() *state {
	return newShake256Generic()
}
