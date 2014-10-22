// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Supply environ, __progname and __guard_local, because
// we don't link against the standard OpenBSD crt0.o and
// the libc dynamic library needs them.

char *environ[1];
char *__progname;
long __guard_local;

#pragma dynexport environ environ
#pragma dynexport __progname __progname

// This is normally marked as hidden and placed in the
// .openbsd.randomdata section.
#pragma dynexport __guard_local __guard_local

// We override pthread_create to support PT_TLS.
#pragma dynexport pthread_create pthread_create
