// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define N SigNotify
#define K SigKill
#define T SigThrow
#define P SigPanic
#define E SigGoExit

// Incoming notes are compared against this table using strncmp, so the
// order matters: longer patterns must appear before their prefixes.
// There are #defined SIG constants in os_plan9.h for the table index of
// some of these.
//
// If you add entries to this table, you must respect the prefix ordering
// and also update the constant values is os_plan9.h.

#pragma dataflag NOPTR
SigTab runtime·sigtab[] = {
	// Traps that we cannot be recovered.
	T,	"sys: trap: debug exception",
	T,	"sys: trap: invalid opcode",

	// We can recover from some memory errors in runtime·sigpanic.
	P,	"sys: trap: fault read addr",	// SIGRFAULT
	P,	"sys: trap: fault write addr",	// SIGWFAULT

	// We can also recover from math errors.
	P,	"sys: trap: divide error",	// SIGINTDIV
	P,	"sys: fp:",	// SIGFLOAT

	// All other traps are normally handled as if they were marked SigThrow.
	// We mark them SigPanic here so that debug.SetPanicOnFault will work.
	P,	"sys: trap:",	// SIGTRAP

	// Writes to a closed pipe can be handled if desired, otherwise they're ignored.
	N,	"sys: write on closed pipe",

	// Other system notes are more serious and cannot be recovered.
	T,	"sys:",

	// Issued to all other procs when calling runtime·exit.
	E,	"go: exit ",

	// Kill is sent by external programs to cause an exit.
	K,	"kill",

	// Interrupts can be handled if desired, otherwise they cause an exit.
	N+K,	"interrupt",
	N+K,	"hangup",

	// Alarms can be handled if desired, otherwise they're ignored.
	N,	"alarm",
};

#undef N
#undef K
#undef T
#undef P
#undef E
