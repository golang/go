// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !gc || purego || !s390x || !ignore

package sha3

func new224() *Digest {
	return new224Generic()
}

func new256() *Digest {
	return new256Generic()
}

func new384() *Digest {
	return new384Generic()
}

func new512() *Digest {
	return new512Generic()
}

func newShake128() *SHAKE {
	return newShake128Generic()
}

func newShake256() *SHAKE {
	return newShake256Generic()
}
