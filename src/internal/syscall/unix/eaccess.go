// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package unix

func Eaccess(path string, mode uint32) error {
	return faccessat(AT_FDCWD, path, mode, AT_EACCESS)
}
