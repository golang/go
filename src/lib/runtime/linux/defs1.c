// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs
	godefs -f -m64 defs.c >amd64/defs.h
	godefs -f -m64 defs1.c >>amd64/defs.h
 */

#include <ucontext.h>

typedef __sigset_t $Usigset;
typedef struct _libc_fpxreg $Fpxreg;
typedef struct _libc_xmmreg $Xmmreg;
typedef struct _libc_fpstate $Fpstate;
typedef struct _libc_fpreg $Fpreg;
typedef struct _fpxreg $Fpxreg1;
typedef struct _xmmreg $Xmmreg1;
typedef struct _fpstate $Fpstate1;
typedef struct _fpreg $Fpreg1;
typedef struct sigaltstack $Sigaltstack;
typedef mcontext_t $Mcontext;
typedef ucontext_t $Ucontext;
typedef struct sigcontext $Sigcontext;
