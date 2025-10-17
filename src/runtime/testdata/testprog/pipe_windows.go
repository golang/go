// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "syscall"

func pipe() (r, w syscall.Handle, err error) {
	var p [2]syscall.Handle
	err = syscall.Pipe(p[:])
	return p[0], p[1], err
}
