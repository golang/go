// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs
	godefs -f -m64 defs.c >amd64/defs.h
	godefs -f -m64 defs1.c >>amd64/defs.h
 */

// Linux glibc and Linux kernel define different and conflicting
// definitions for struct sigaction, struct timespec, etc.
// We want the kernel ones, which are in the asm/* headers.
// But then we'd get conflicts when we include the system
// headers for things like ucontext_t, so that happens in
// a separate file, defs1.c.

#include <asm/signal.h>
#include <asm/siginfo.h>
#include <asm/mman.h>

enum {
	$PROT_NONE = PROT_NONE,
	$PROT_READ = PROT_READ,
	$PROT_WRITE = PROT_WRITE,
	$PROT_EXEC = PROT_EXEC,

	$MAP_ANON = MAP_ANONYMOUS,
	$MAP_PRIVATE = MAP_PRIVATE,

	$SA_RESTART = SA_RESTART,
	$SA_ONSTACK = SA_ONSTACK,
	$SA_RESTORER = SA_RESTORER,
	$SA_SIGINFO = SA_SIGINFO,
};

typedef struct timespec $Timespec;
typedef struct timeval $Timeval;
typedef struct sigaction $Sigaction;
typedef siginfo_t $Siginfo;
