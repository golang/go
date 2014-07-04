// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !android,linux

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "libcgo.h"

void
fatalf(const char* format, ...)
{
	va_list ap;

	fprintf(stderr, "runtime/cgo: ");
	va_start(ap, format);
	vfprintf(stderr, format, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	abort();
}
