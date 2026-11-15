// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

/*
#include <errno.h>

void bench_void_fast(void) {}
void bench_void_slow(void) {}

int bench_one_fast(int a) { return a + 1; }
int bench_one_slow(int a) { return a + 1; }

__attribute__((noinline)) int bench_three_fast(int a, int b, int c) { return a + b + c; }
__attribute__((noinline)) int bench_three_slow(int a, int b, int c) { return a + b + c; }

int bench_read2_fast(int *a, int *b) { return *a + *b; }
int bench_read2_slow(int *a, int *b) { return *a + *b; }

int bench_sum2_fast(unsigned char *p, int n, unsigned char *q, int m) {
	int s = 0;
	for (int i = 0; i < n; i++) {
		s += p[i];
	}
	for (int i = 0; i < m; i++) {
		s += q[i];
	}
	return s;
}
int bench_sum2_slow(unsigned char *p, int n, unsigned char *q, int m) {
	int s = 0;
	for (int i = 0; i < n; i++) {
		s += p[i];
	}
	for (int i = 0; i < m; i++) {
		s += q[i];
	}
	return s;
}

int bench_mayfail_fast(int fail) {
	if (fail) {
		errno = EINVAL;
		return -1;
	}
	return 0;
}
int bench_mayfail_slow(int fail) {
	if (fail) {
		errno = EINVAL;
		return -1;
	}
	return 0;
}

typedef struct {
	int x;
	int y;
	long long z;
} bench_point_t;

int bench_point_fast(bench_point_t p) { return p.x + p.y + (int)p.z; }
int bench_point_slow(bench_point_t p) { return p.x + p.y + (int)p.z; }

int bench_burn_fast(int n) {
	int s = 0;
	for (int i = 0; i < n; i++) {
		s += (i ^ (i >> 3)) & 15;
	}
	return s;
}
int bench_burn_slow(int n) {
	int s = 0;
	for (int i = 0; i < n; i++) {
		s += (i ^ (i >> 3)) & 15;
	}
	return s;
}

long long bench_floor_empty(long long iters) {
	long long s = 0;
	for (long long i = 0; i < iters; i++) {
		s += (s & 7) + 3;
	}
	return s;
}
long long bench_floor_call(long long iters) {
	long long s = 0;
	for (long long i = 0; i < iters; i++) {
		s += bench_three_fast((int)(s & 7), 1, 2);
	}
	return s;
}

#cgo noescape bench_void_fast
#cgo nocallback bench_void_fast
#cgo noescape bench_one_fast
#cgo nocallback bench_one_fast
#cgo noescape bench_three_fast
#cgo nocallback bench_three_fast
#cgo noescape bench_read2_fast
#cgo nocallback bench_read2_fast
#cgo noescape bench_sum2_fast
#cgo nocallback bench_sum2_fast
#cgo noescape bench_mayfail_fast
#cgo nocallback bench_mayfail_fast
#cgo noescape bench_point_fast
#cgo nocallback bench_point_fast
#cgo noescape bench_burn_fast
#cgo nocallback bench_burn_fast
#cgo noescape bench_void_slow
#cgo noescape bench_one_slow
#cgo noescape bench_three_slow
#cgo noescape bench_read2_slow
#cgo noescape bench_sum2_slow
#cgo noescape bench_mayfail_slow
#cgo noescape bench_point_slow
#cgo noescape bench_burn_slow
*/
import "C"

import "unsafe"

func callBenchVoidFast() { C.bench_void_fast() }

func callBenchVoidSlow() { C.bench_void_slow() }

func callBenchOneFast(a int32) int32 { return int32(C.bench_one_fast(C.int(a))) }

func callBenchOneSlow(a int32) int32 { return int32(C.bench_one_slow(C.int(a))) }

func callBenchThreeFast(a, b, c int32) int32 {
	return int32(C.bench_three_fast(C.int(a), C.int(b), C.int(c)))
}

func callBenchThreeSlow(a, b, c int32) int32 {
	return int32(C.bench_three_slow(C.int(a), C.int(b), C.int(c)))
}

func callBenchRead2Fast(a, b *int32) int32 {
	return int32(C.bench_read2_fast((*C.int)(unsafe.Pointer(a)), (*C.int)(unsafe.Pointer(b))))
}

func callBenchRead2Slow(a, b *int32) int32 {
	return int32(C.bench_read2_slow((*C.int)(unsafe.Pointer(a)), (*C.int)(unsafe.Pointer(b))))
}

func callBenchSum2Fast(p, q []byte) int32 {
	return int32(C.bench_sum2_fast(
		(*C.uchar)(unsafe.Pointer(&p[0])), C.int(len(p)),
		(*C.uchar)(unsafe.Pointer(&q[0])), C.int(len(q))))
}

func callBenchSum2Slow(p, q []byte) int32 {
	return int32(C.bench_sum2_slow(
		(*C.uchar)(unsafe.Pointer(&p[0])), C.int(len(p)),
		(*C.uchar)(unsafe.Pointer(&q[0])), C.int(len(q))))
}

func callBenchMayfailFast(fail int32) (int32, error) {
	r, err := C.bench_mayfail_fast(C.int(fail))
	return int32(r), err
}

func callBenchMayfailSlow(fail int32) (int32, error) {
	r, err := C.bench_mayfail_slow(C.int(fail))
	return int32(r), err
}

func callBenchPointFast(p C.bench_point_t) int32 { return int32(C.bench_point_fast(p)) }

func callBenchPointSlow(p C.bench_point_t) int32 { return int32(C.bench_point_slow(p)) }

func benchPointVals() (C.bench_point_t, C.bench_point_t) {
	return C.bench_point_t{x: 1, y: 2, z: 3}, C.bench_point_t{x: 1, y: 2, z: 3}
}

func callBenchBurnFast(n int32) int32 { return int32(C.bench_burn_fast(C.int(n))) }

func callBenchBurnSlow(n int32) int32 { return int32(C.bench_burn_slow(C.int(n))) }

func callBenchFloorEmpty(iters int64) int64 { return int64(C.bench_floor_empty(C.longlong(iters))) }

func callBenchFloorCall(iters int64) int64 { return int64(C.bench_floor_call(C.longlong(iters))) }
