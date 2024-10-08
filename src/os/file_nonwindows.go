// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package os

import "syscall"

func mkdir(name string, perm FileMode) error {
	return ignoringEINTR(func() error {
		return syscall.Mkdir(name, syscallMode(perm))
	})
}
