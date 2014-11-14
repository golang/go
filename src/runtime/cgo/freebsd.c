// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd

#include "textflag.h"

// Supply environ and __progname, because we don't
// link against the standard FreeBSD crt0.o and the
// libc dynamic library needs them.

#pragma dataflag NOPTR
char *environ[1];
#pragma dataflag NOPTR
char *__progname;

#pragma dynexport environ environ
#pragma dynexport __progname __progname
