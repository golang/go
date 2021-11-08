// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

#ifndef _GNU_SOURCE // setres[ug]id() API.
#define _GNU_SOURCE
#endif

#include <grp.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include "libcgo.h"

/*
 * Assumed POSIX compliant libc system call wrappers. For linux, the
 * glibc/nptl/setxid mechanism ensures that POSIX semantics are
 * honored for all pthreads (by default), and this in turn with cgo
 * ensures that all Go threads launched with cgo are kept in sync for
 * these function calls.
 */

// argset_t matches runtime/cgocall.go:argset.
typedef struct {
	uintptr_t* args;
	uintptr_t retval;
} argset_t;

// libc backed posix-compliant syscalls.

#define SET_RETVAL(fn) \
  uintptr_t ret = (uintptr_t) fn ; \
  if (ret == (uintptr_t) -1) {	   \
    x->retval = (uintptr_t) errno; \
  } else                           \
    x->retval = ret

void
_cgo_libc_setegid(argset_t* x) {
	SET_RETVAL(setegid((gid_t) x->args[0]));
}

void
_cgo_libc_seteuid(argset_t* x) {
	SET_RETVAL(seteuid((uid_t) x->args[0]));
}

void
_cgo_libc_setgid(argset_t* x) {
	SET_RETVAL(setgid((gid_t) x->args[0]));
}

void
_cgo_libc_setgroups(argset_t* x) {
	SET_RETVAL(setgroups((size_t) x->args[0], (const gid_t *) x->args[1]));
}

void
_cgo_libc_setregid(argset_t* x) {
	SET_RETVAL(setregid((gid_t) x->args[0], (gid_t) x->args[1]));
}

void
_cgo_libc_setresgid(argset_t* x) {
	SET_RETVAL(setresgid((gid_t) x->args[0], (gid_t) x->args[1],
			     (gid_t) x->args[2]));
}

void
_cgo_libc_setresuid(argset_t* x) {
	SET_RETVAL(setresuid((uid_t) x->args[0], (uid_t) x->args[1],
			     (uid_t) x->args[2]));
}

void
_cgo_libc_setreuid(argset_t* x) {
	SET_RETVAL(setreuid((uid_t) x->args[0], (uid_t) x->args[1]));
}

void
_cgo_libc_setuid(argset_t* x) {
	SET_RETVAL(setuid((uid_t) x->args[0]));
}
