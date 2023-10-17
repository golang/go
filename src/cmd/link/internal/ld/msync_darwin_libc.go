// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package ld

import _ "unsafe" // for go:linkname

//go:linkname msync syscall.msync
func msync(b []byte, flags int) (err error)
