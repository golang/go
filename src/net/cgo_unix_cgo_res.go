// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// res_search, for cgo systems where that is thread-safe.

//go:build cgo && !netgo && (linux || openbsd)

package net

/*
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <arpa/nameser.h>
#include <resolv.h>
#include <netdb.h>

#ifdef __GLIBC__
#define is_glibc 1
#else
#define is_glibc 0
#endif

int c_res_search(const char *dname, int class, int type, unsigned char *answer, int anslen, int *herrno) {
	int ret = res_search(dname, class, type, answer, anslen);

	if (ret < 0) {
		*herrno = h_errno;
	}

	return ret;
}

#cgo !android,!openbsd LDFLAGS: -lresolv
*/
import "C"

import "runtime"

type _C_struct___res_state = struct{}

func _C_res_ninit(state *_C_struct___res_state) error {
	return nil
}

func _C_res_nclose(state *_C_struct___res_state) {
	return
}

const isGlibc = C.is_glibc == 1

func _C_res_nsearch(state *_C_struct___res_state, dname *_C_char, class, typ int, ans *_C_uchar, anslen int) (ret int, herrno int, err error) {
	var h C.int
	x, err := C.c_res_search(dname, C.int(class), C.int(typ), ans, C.int(anslen), &h)

	if x <= 0 {
		if runtime.GOOS == "linux" {
			// On glibc and musl h_errno is a thread-safe macro:
			// https://sourceware.org/git/?p=glibc.git;a=commitdiff;h=ed4d0184a16db455d626e64722daf9ca4b71742a;hp=c0b338331e8eb0979b909479d6aa9fd1cddd63ec
			// http://git.etalabs.net/cgit/musl/commit/?id=9d0b8b92a508c328e7eac774847f001f80dfb5ff
			if isGlibc {
				return -1, int(h), err
			}
			// musl does not set errno with the cause of the failure.
			return -1, int(h), nil
		}

		// On Openbsd h_errno is not thread-safe.
		// Android h_errno is also thread-safe: https://android.googlesource.com/platform/bionic/+/589afca/libc/dns/resolv/res_state.c
		// but the h_errno doesn't seem to be set on noSuchHost.
		return -2, 0, nil
	}

	return int(x), 0, nil
}
