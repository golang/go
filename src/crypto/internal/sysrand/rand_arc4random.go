// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || openbsd

package sysrand

import "internal/syscall/unix"

// arc4random_buf is the recommended application CSPRNG, accepts buffers of
// any size, and never returns an error.
//
// "The subsystem is re-seeded from the kernel random number subsystem on a
// regular basis, and also upon fork(2)." - arc4random(3)
//
// Note that despite its legacy name, it uses a secure CSPRNG (not RC4) in
// all supported macOS versions.
func read(b []byte) error {
	unix.ARC4Random(b)
	return nil
}
