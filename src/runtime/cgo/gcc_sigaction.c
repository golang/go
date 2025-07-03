// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (amd64 || arm64 || loong64 || ppc64le)

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>

#include "libcgo.h"

// go_sigaction_t is a C version of the sigactiont struct from
// defs_${goos}_${goarch}.go.  This definition — and its conversion
// to and from struct sigaction — are specific to ${goos}/${goarch}.
typedef struct {
	uintptr_t handler;
	uint64_t flags;
#ifdef __loongarch__
	uint64_t mask;
	uintptr_t restorer;
#else
	uintptr_t restorer;
	uint64_t mask;
#endif
} go_sigaction_t;

// SA_RESTORER is part of the kernel interface.
// This is Linux i386/amd64 specific.
#ifndef SA_RESTORER
#define SA_RESTORER 0x4000000
#endif

int32_t
x_cgo_sigaction(intptr_t signum, const go_sigaction_t *goact, go_sigaction_t *oldgoact) {
	int32_t ret;
	struct sigaction act;
	struct sigaction oldact;
	size_t i;

	_cgo_tsan_acquire();

	memset(&act, 0, sizeof act);
	memset(&oldact, 0, sizeof oldact);

	if (goact) {
		if (goact->flags & SA_SIGINFO) {
			act.sa_sigaction = (void(*)(int, siginfo_t*, void*))(goact->handler);
		} else {
			act.sa_handler = (void(*)(int))(goact->handler);
		}
		sigemptyset(&act.sa_mask);
		for (i = 0; i < 8 * sizeof(goact->mask); i++) {
			if (goact->mask & ((uint64_t)(1)<<i)) {
				sigaddset(&act.sa_mask, (int)(i+1));
			}
		}
		act.sa_flags = (int)(goact->flags & ~(uint64_t)SA_RESTORER);
	}

	ret = sigaction((int)signum, goact ? &act : NULL, oldgoact ? &oldact : NULL);
	if (ret == -1) {
		// runtime.rt_sigaction expects _cgo_sigaction to return errno on error.
		_cgo_tsan_release();
		return errno;
	}

	if (oldgoact) {
		if (oldact.sa_flags & SA_SIGINFO) {
			oldgoact->handler = (uintptr_t)(oldact.sa_sigaction);
		} else {
			oldgoact->handler = (uintptr_t)(oldact.sa_handler);
		}
		oldgoact->mask = 0;
		for (i = 0; i < 8 * sizeof(oldgoact->mask); i++) {
			if (sigismember(&oldact.sa_mask, (int)(i+1)) == 1) {
				oldgoact->mask |= (uint64_t)(1)<<i;
			}
		}
		oldgoact->flags = (uint64_t)oldact.sa_flags;
	}

	_cgo_tsan_release();
	return ret;
}
