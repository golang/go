// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

package boring

import (
	"hash"
)

type sha interface {
	NewSHA1() hash.Hash
	NewSHA224() hash.Hash
	NewSHA256() hash.Hash
	NewSHA384() hash.Hash
	NewSHA512() hash.Hash
}

// NewSHA1 returns a new SHA1 hash.
func NewSHA1() hash.Hash {
	return external.NewSHA1()
}

// NewSHA224 returns a new SHA224 hash.
func NewSHA224() hash.Hash {
	return external.NewSHA224()
}

// NewSHA256 returns a new SHA256 hash.
func NewSHA256() hash.Hash {
	return external.NewSHA256()
}

// NewSHA384 returns a new SHA384 hash.
func NewSHA384() hash.Hash {
	return external.NewSHA384()
}

// NewSHA512 returns a new SHA512 hash.
func NewSHA512() hash.Hash {
	return external.NewSHA512()
}
