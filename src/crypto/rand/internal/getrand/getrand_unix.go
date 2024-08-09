// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || dragonfly || freebsd || illumos || solaris

package getrand

import (
	"internal/syscall/unix"
	"syscall"
)

func getRandom(out []byte) error {
	n, err := unix.GetRandom(out, 0)
	if err != nil {
		return err
	}
	if n != len(out) {
		return syscall.EIO
	}
	return nil
}
