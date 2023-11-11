// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package unix

import "syscall"

func Fcntl(fd int, cmd int, arg int) (int, error) {
	return 0, syscall.ENOSYS
}
