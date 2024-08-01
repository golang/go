// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd || netbsd

package rand

import "internal/syscall/unix"

func read(b []byte) error {
	for len(b) > 0 {
		size := len(b)
		if size > 256 {
			size = 256
		}
		// getentropy(2) returns a maximum of 256 bytes per call.
		if err := unix.GetEntropy(b[:size]); err != nil {
			return err
		}
		b = b[size:]
	}
	return nil
}
