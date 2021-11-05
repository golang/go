// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (darwin && !ios) || openbsd

package rand

import (
	"internal/syscall/unix"
)

func init() {
	altGetRandom = getEntropy
}

func getEntropy(p []byte) (ok bool) {
	// getentropy(2) returns a maximum of 256 bytes per call
	for i := 0; i < len(p); i += 256 {
		end := i + 256
		if len(p) < end {
			end = len(p)
		}
		err := unix.GetEntropy(p[i:end])
		if err != nil {
			return false
		}
	}
	return true
}
