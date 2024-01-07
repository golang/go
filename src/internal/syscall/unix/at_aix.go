// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

//go:cgo_import_dynamic libc_fstatat fstatat "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_openat openat "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_unlinkat unlinkat "libc.a/shr_64.o"

const (
	AT_REMOVEDIR        = 0x1
	AT_SYMLINK_NOFOLLOW = 0x1
	UTIME_OMIT          = -0x3
)
