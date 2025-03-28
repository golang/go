// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (unix && !dragonfly && !freebsd && !netbsd) || wasip1

package unix

import "syscall"

// POSIX.1-2008 says it's ELOOP. Most platforms follow:
//
//   - aix: O_NOFOLLOW not documented (https://www.ibm.com/docs/ssw_aix_73/o_bostechref/open.html), assuming ELOOP
//   - android: see linux
//   - darwin: https://github.com/apple/darwin-xnu/blob/main/bsd/man/man2/open.2
//   - hurd: who knows if it works at all (https://www.gnu.org/software/hurd/open_issues/open_symlink.html)
//   - illumos: https://illumos.org/man/2/open
//   - ios: see darwin
//   - linux: https://man7.org/linux/man-pages/man2/openat.2.html
//   - openbsd: https://man.openbsd.org/open.2
//   - solaris: https://docs.oracle.com/cd/E23824_01/html/821-1463/open-2.html
const noFollowErrno = syscall.ELOOP
