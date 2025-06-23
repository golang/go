// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(darwin || freebsd || linux || netbsd || openbsd)

package toolchain

import "io/fs"

func sysWriteBits() fs.FileMode {
	return 0700
}
