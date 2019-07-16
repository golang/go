// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package runtime

func sigismember(mask *sigset, i int) bool {
	clear := *mask
	sigdelset(&clear, i)
	return clear != *mask
}

func Sigisblocked(i int) bool {
	var sigmask sigset
	sigprocmask(_SIG_SETMASK, nil, &sigmask)
	return sigismember(&sigmask, i)
}
