// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netbsd

#include <string.h>
#include <signal.h>

static void
threadentry_platform(void)
{
	// On NetBSD, a new thread inherits the signal stack of the
	// creating thread. That confuses minit, so we remove that
	// signal stack here before calling the regular mstart. It's
	// a bit baroque to remove a signal stack here only to add one
	// in minit, but it's a simple change that keeps NetBSD
	// working like other OS's. At this point all signals are
	// blocked, so there is no race.
	stack_t ss;
	memset(&ss, 0, sizeof ss);
	ss.ss_flags = SS_DISABLE;
	sigaltstack(&ss, NULL);
}

void (*x_cgo_threadentry_platform)(void) = threadentry_platform;
