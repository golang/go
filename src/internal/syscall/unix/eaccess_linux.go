// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "syscall"

func Eaccess(path string, mode uint32) error {
	return syscall.Faccessat(AT_FDCWD, path, mode, AT_EACCESS)
}
