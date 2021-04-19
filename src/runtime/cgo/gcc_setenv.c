// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

#include "libcgo.h"

#include <stdlib.h>

/* Stub for calling setenv */
void
x_cgo_setenv(char **arg)
{
	_cgo_tsan_acquire();
	setenv(arg[0], arg[1], 1);
	_cgo_tsan_release();
}

/* Stub for calling unsetenv */
void
x_cgo_unsetenv(char *arg)
{
	_cgo_tsan_acquire();
	unsetenv(arg);
	_cgo_tsan_release();
}
