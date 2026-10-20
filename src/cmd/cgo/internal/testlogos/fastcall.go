// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testlogos exercises the M20-S1 fast-cgo path: C functions
// carrying the #cgo noescape plus #cgo nocallback pair (or the .lgo
// spelling //logos:cgo_fastcall, covered in testdata) take the fast
// call path through runtime.logosCgocallFast instead of runtime.cgocall.
// Functions carrying only one of the two directives keep the plain path.
package testlogos

/*
#include <stdint.h>
#include <time.h>

// The file carries //export, so the preamble must hold declarations
// only (cmd/cgo/doc.go): the bodies live in fastcall_c.c.

int fast_add(int a, int b);
int slow_noescape(int a, int b);
int slow_nocallback(int a, int b);
int block_sleep(int micros);
int plain_sleep(int micros);
int read_int(int *p);
int sum_bytes(unsigned char *p, int n);
unsigned long long busy_spin(unsigned long long iters);
unsigned long long plain_spin(unsigned long long iters);
extern int GoCallback(int v);
int cb_fire(void);
int plain_cb_fire(void);

#cgo noescape fast_add
#cgo nocallback fast_add

#cgo noescape slow_noescape

#cgo nocallback slow_nocallback

#cgo noescape block_sleep
#cgo nocallback block_sleep

#cgo noescape read_int
#cgo nocallback read_int

#cgo noescape sum_bytes
#cgo nocallback sum_bytes

#cgo noescape busy_spin
#cgo nocallback busy_spin

#cgo noescape cb_fire
#cgo nocallback cb_fire

#cgo nocallback plain_cb_fire
*/
import "C"

import "unsafe"

//export GoCallback
func GoCallback(v C.int) C.int { return v + 1 }

// callFastAdd routes through the fast entry when the experiment is on
// and falls back to the plain entry otherwise.
func callFastAdd(a, b int32) int32 { return int32(C.fast_add(C.int(a), C.int(b))) }

// callSlowNoescape carries only noescape and keeps the plain entry.
func callSlowNoescape(a, b int32) int32 { return int32(C.slow_noescape(C.int(a), C.int(b))) }

// callSlowNocallback carries only nocallback and keeps the plain entry.
func callSlowNocallback(a, b int32) int32 {
	return int32(C.slow_nocallback(C.int(a), C.int(b)))
}

// callFastSleep blocks in C on the fast path.
func callFastSleep(micros int32) int32 { return int32(C.block_sleep(C.int(micros))) }

// callPlainSleep blocks in C on the plain path.
func callPlainSleep(micros int32) int32 { return int32(C.plain_sleep(C.int(micros))) }

// callFastRead passes a Go pointer through the fast path.
func callFastRead(p *int32) int32 { return int32(C.read_int((*C.int)(unsafe.Pointer(p)))) }

// callFastSlice passes a primitive slice through the fast path.
func callFastSlice(b []byte) int32 {
	if len(b) == 0 {
		return 0
	}
	return int32(C.sum_bytes((*C.uchar)(unsafe.Pointer(&b[0])), C.int(len(b))))
}

// callFastBusy burns CPU in C on the fast path.
func callFastBusy(iters uint64) uint64 { return uint64(C.busy_spin(C.ulonglong(iters))) }

// callPlainBusy burns CPU in C on the plain path.
func callPlainBusy(iters uint64) uint64 { return uint64(C.plain_spin(C.ulonglong(iters))) }

// callFastCallback invokes a C function that calls back into Go.
func callFastCallback() int32 { return int32(C.cb_fire()) }

// callPlainCallback invokes a plain-path C function that calls back
// into Go.
func callPlainCallback() int32 { return int32(C.plain_cb_fire()) }
