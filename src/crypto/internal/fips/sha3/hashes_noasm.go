// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !gc || purego || !s390x

package sha3

func new224() *state {
	return new224Generic()
}

func new256() *state {
	return new256Generic()
}

func new384() *state {
	return new384Generic()
}

func new512() *state {
	return new512Generic()
}
