// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd && amd64

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>

#include "libcgo.h"

// go_sigaction_t is a C version of the sigactiont struct from
// os_freebsd.go.  This definition — and its conversion to and from struct
// sigaction — are specific to freebsd/amd64.
typedef struct {
        uint32_t __bits[_SIG_WORDS];
} go_sigset_t;
typedef struct {
	uintptr_t handler;
	int32_t flags;
	go_sigset_t mask;
} go_sigaction_t;

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
			if (goact->mask.__bits[i/32] & ((uint32_t)(1)<<(i&31))) {
				sigaddset(&act.sa_mask, i+1);
			}
		}
		act.sa_flags = goact->flags;
	}

	ret = sigaction(signum, goact ? &act : NULL, oldgoact ? &oldact : NULL);
	if (ret == -1) {
		// runtime.sigaction expects _cgo_sigaction to return errno on error.
		_cgo_tsan_release();
		return errno;
	}

	if (oldgoact) {
		if (oldact.sa_flags & SA_SIGINFO) {
			oldgoact->handler = (uintptr_t)(oldact.sa_sigaction);
		} else {
			oldgoact->handler = (uintptr_t)(oldact.sa_handler);
		}
		for (i = 0 ; i < _SIG_WORDS; i++) {
			oldgoact->mask.__bits[i] = 0;
		}
		for (i = 0; i < 8 * sizeof(oldgoact->mask); i++) {
			if (sigismember(&oldact.sa_mask, i+1) == 1) {
				oldgoact->mask.__bits[i/32] |= (uint32_t)(1)<<(i&31);
			}
		}
		oldgoact->flags = oldact.sa_flags;
	}

	_cgo_tsan_release();
	return ret;
}
