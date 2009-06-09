// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs
	godefs -carm-gcc -f -I/usr/local/google/src/linux-2.6.28/arch/arm/include -f
 -I/usr/local/google/src/linux-2.6.28/include defs_arm.c >arm/defs.h

 * Another input file for ARM defs.h
 */

#include <asm/signal.h>
#include <asm/mman.h>
#include <asm/sigcontext.h>
#include <asm/ucontext.h>

/*
#include <sys/signal.h>
#include <sys/mman.h>
#include <ucontext.h>
*/

#include <time.h>

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
	$SA_SIGINFO = SA_SIGINFO
};




//typedef struct _fpreg $Fpreg;
//typedef struct _fpxreg $Fpxreg;
//typedef struct _xmmreg $Xmmreg;
//typedef struct _fpstate $Fpstate;
typedef struct timespec $Timespec;
//typedef struct timeval $Timeval;
// typedef struct sigaction $Sigaction;
// typedef siginfo_t $Siginfo;
// typedef struct sigaltstack $Sigaltstack;
// typedef struct sigcontext $Sigcontext;
// typedef struct ucontext $Ucontext;
