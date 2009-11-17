// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs.
 *
	godefs -f -m64 defs.c >amd64/defs.h
	godefs defs.c >386/defs.h
 */

#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <sys/umtx.h>
#include <sys/_sigset.h>

enum {
	$PROT_NONE = PROT_NONE,
	$PROT_READ = PROT_READ,
	$PROT_WRITE = PROT_WRITE,
	$PROT_EXEC = PROT_EXEC,

	$MAP_ANON = MAP_ANON,
	$MAP_PRIVATE = MAP_PRIVATE,

	$SA_SIGINFO = SA_SIGINFO,
	$SA_RESTART = SA_RESTART,
	$SA_ONSTACK = SA_ONSTACK,

	$UMTX_OP_WAIT = UMTX_OP_WAIT,
	$UMTX_OP_WAKE = UMTX_OP_WAKE,

	$EINTR = EINTR,
};

typedef struct sigaltstack $Sigaltstack;
typedef struct __sigset $Sigset;
typedef union sigval $Sigval;
typedef stack_t	$StackT;

typedef siginfo_t $Siginfo;

typedef mcontext_t $Mcontext;
typedef ucontext_t $Ucontext;
typedef struct sigcontext $Sigcontext;
