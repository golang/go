// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

package unix

import _ "unsafe" // for linkname

// GetEntropy calls the OpenBSD getentropy system call.
func GetEntropy(p []byte) error {
	return getentropy(p)
}

//go:linkname getentropy syscall.getentropy
//go:noescape
func getentropy(p []byte) error
