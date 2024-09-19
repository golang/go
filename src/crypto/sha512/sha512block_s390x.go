// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha512

import "internal/cpu"

//go:noescape
func blockS390X(dig *digest, p []byte)

func block(dig *digest, p []byte) {
	if cpu.S390X.HasSHA512 {
		blockS390X(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
