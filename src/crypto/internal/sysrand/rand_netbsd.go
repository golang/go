// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sysrand

import "internal/syscall/unix"

func read(b []byte) error {
	for len(b) > 0 {
		size := len(b)
		// "Returns independent uniformly distributed bytes at random each time,
		// as many as requested up to 256, derived from the system entropy pool;
		// see rnd(4)." -- man sysctl(7)
		if size > 256 {
			size = 256
		}
		if err := unix.Arandom(b[:size]); err != nil {
			return err
		}
		b = b[size:]
	}
	return nil
}
