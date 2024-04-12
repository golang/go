// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd || netbsd

package getrand

import (
	"internal/syscall/unix"
)

// getentropy(2) returns a maximum of 256 bytes per call.
const maxGetRandomRead = 256

func getRandom(out []byte) error {
	return unix.GetEntropy(out)
}
