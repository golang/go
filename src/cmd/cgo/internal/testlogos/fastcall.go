// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

/*
#include <stdint.h>
#include <time.h>

int fast_add(int a, int b);
int slow_noescape(int a, int b);
int slow_nocallback(int a, int b);
int block_sleep(int micros);
int plain_sleep(int micros);
int read_int(int *p);
int sum_bytes(unsigned char *p, int n);
unsigned long long busy_spin(unsigned long long iters);
unsigned long long plain_spin(unsigned long long iters);
int gate_wait(int ready_fd, int gate_fd);
int gate_watch(int ready_fd, int gate_fd, unsigned char *p, unsigned char want);

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

#cgo noescape gate_wait
#cgo nocallback gate_wait

#cgo noescape gate_watch
#cgo nocallback gate_watch
*/
import "C"

import "unsafe"

func callFastAdd(a, b int32) int32 { return int32(C.fast_add(C.int(a), C.int(b))) }

func callSlowNoescape(a, b int32) int32 { return int32(C.slow_noescape(C.int(a), C.int(b))) }

func callSlowNocallback(a, b int32) int32 {
	return int32(C.slow_nocallback(C.int(a), C.int(b)))
}

func callFastSleep(micros int32) int32 { return int32(C.block_sleep(C.int(micros))) }

func callPlainSleep(micros int32) int32 { return int32(C.plain_sleep(C.int(micros))) }

func callFastRead(p *int32) int32 { return int32(C.read_int((*C.int)(unsafe.Pointer(p)))) }

func callFastSlice(b []byte) int32 {
	if len(b) == 0 {
		return 0
	}
	return int32(C.sum_bytes((*C.uchar)(unsafe.Pointer(&b[0])), C.int(len(b))))
}

func callFastBusy(iters uint64) uint64 { return uint64(C.busy_spin(C.ulonglong(iters))) }

func callPlainBusy(iters uint64) uint64 { return uint64(C.plain_spin(C.ulonglong(iters))) }

func callFastGateWait(readyFD, gateFD int32) int32 {
	return int32(C.gate_wait(C.int(readyFD), C.int(gateFD)))
}

func callFastGateWatch(readyFD, gateFD int32, p *byte, want byte) int32 {
	return int32(C.gate_watch(C.int(readyFD), C.int(gateFD), (*C.uchar)(unsafe.Pointer(p)), C.uchar(want)))
}
