// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stddef.h>

#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)

// Mingw seems not to have threads.h, so we use the _Thread_local keyword rather
// than the thread_local macro.
static _Thread_local int tls;

const char *
checkTLS() {
	return NULL;
}

void
setTLS(int v)
{
	tls = v;
}

int
getTLS()
{
	return tls;
}

#else

const char *
checkTLS() {
	return "_Thread_local requires C11 and not __STDC_NO_THREADS__";
}

void
setTLS(int v) {
}

int
getTLS()
{
	return 0;
}

#endif
