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

#cgo !android,!openbsd LDFLAGS: -lresolv
*/
import "C"

type _C_struct___res_state = struct{}

func _C_res_ninit(state *_C_struct___res_state) error {
	return nil
}

func _C_res_nclose(state *_C_struct___res_state) {
	return
}

func _C_res_nsearch(state *_C_struct___res_state, dname *_C_char, class, typ int, ans *_C_uchar, anslen int) (int, error) {
	x, err := C.res_search(dname, C.int(class), C.int(typ), ans, C.int(anslen))
	return int(x), err
}
