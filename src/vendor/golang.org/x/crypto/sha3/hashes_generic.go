// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !gc || purego || !s390x

package sha3

import (
	"hash"
)

// new224Asm returns an assembly implementation of SHA3-224 if available,
// otherwise it returns nil.
func new224Asm() hash.Hash { return nil }

// new256Asm returns an assembly implementation of SHA3-256 if available,
// otherwise it returns nil.
func new256Asm() hash.Hash { return nil }

// new384Asm returns an assembly implementation of SHA3-384 if available,
// otherwise it returns nil.
func new384Asm() hash.Hash { return nil }

// new512Asm returns an assembly implementation of SHA3-512 if available,
// otherwise it returns nil.
func new512Asm() hash.Hash { return nil }
