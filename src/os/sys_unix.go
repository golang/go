// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package os

// supportsCloseOnExec reports whether the platform supports the
// O_CLOEXEC flag.
// On Darwin, the O_CLOEXEC flag was introduced in OS X 10.7 (Darwin 11.0.0).
// See https://support.apple.com/kb/HT1633.
// On FreeBSD, the O_CLOEXEC flag was introduced in version 8.3.
const supportsCloseOnExec = true
