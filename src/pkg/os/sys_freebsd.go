// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

// supportsCloseOnExec reports whether the platform supports the
// O_CLOEXEC flag.
var supportsCloseOnExec bool

func init() {
	osrel, err := syscall.SysctlUint32("kern.osreldate")
	if err != nil {
		return
	}
	// The O_CLOEXEC flag was introduced in FreeBSD 8.3.
	// See http://www.freebsd.org/doc/en/books/porters-handbook/freebsd-versions.html.
	if osrel >= 803000 {
		supportsCloseOnExec = true
	}
}
