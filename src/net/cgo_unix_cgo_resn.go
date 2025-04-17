// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// res_nsearch, for cgo systems where that's available.

//go:build cgo && !netgo && unix && !(darwin || linux || openbsd)

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

#cgo !aix,!dragonfly,!freebsd LDFLAGS: -lresolv
*/
import "C"

type _C_struct___res_state = C.struct___res_state

func _C_res_ninit(state *_C_struct___res_state) error {
	_, err := C.res_ninit(state)
	return err
}

func _C_res_nclose(state *_C_struct___res_state) {
	C.res_nclose(state)
}

func _C_res_nsearch(state *_C_struct___res_state, dname *_C_char, class, typ int, ans *_C_uchar, anslen int) int {
	x := C.res_nsearch(state, dname, C.int(class), C.int(typ), ans, C.int(anslen))
	return int(x)
}
