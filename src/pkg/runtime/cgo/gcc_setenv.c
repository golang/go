// Copyright 20111 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

#include "libcgo.h"

#include <stdlib.h>

/* Stub for calling setenv */
static void
xlibcgo_setenv(char **arg)
{
	setenv(arg[0], arg[1], 1);
}

void (*libcgo_setenv)(char**) = xlibcgo_setenv;
