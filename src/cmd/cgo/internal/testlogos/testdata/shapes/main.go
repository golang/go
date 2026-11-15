// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern int GoCallback(int v);

struct ShapePoint { int x; int y; };

enum ShapeColor { SHAPE_RED = 1, SHAPE_GREEN = 2, SHAPE_BLUE = 3 };

typedef int (*shape_fn_t)(int);

int shape_add_fast(int a, int b);
int shape_add_slow(int a, int b);
unsigned long long shape_u64_fast(unsigned long long a);
unsigned long long shape_u64_slow(unsigned long long a);
float shape_f32_fast(float a);
float shape_f32_slow(float a);
double shape_f64_fast(double a);
double shape_f64_slow(double a);
bool shape_bool_fast(bool b);
bool shape_bool_slow(bool b);
void shape_nop_fast(void);
void shape_nop_slow(void);
void shape_void_fast(int *p);
void shape_void_slow(int *p);
int shape_read_fast(int *p);
int shape_read_slow(int *p);
int shape_ppre_fast(int **pp);
int shape_ppre_slow(int **pp);
int shape_sum_fast(unsigned char *p, int n);
int shape_sum_slow(unsigned char *p, int n);
int shape_point_fast(struct ShapePoint p);
int shape_point_slow(struct ShapePoint p);
int shape_color_fast(enum ShapeColor c);
int shape_color_slow(enum ShapeColor c);
int shape_double(int x);
int shape_apply_fast(shape_fn_t f, int x);
int shape_apply_slow(shape_fn_t f, int x);
int shape_errno_fast(int v);
int shape_errno_slow(int v);
int shape_inline_fast(int x);
int shape_inline_slow(int x);
int cgoCheckPointer(int x);
int cgocall(int x);
int shape_strlen_fast(char *s);
int shape_strlen_slow(char *s);
int shape_vp_fast(void *p);
int shape_vp_slow(void *p);
int shape_cb_fast(void);
int shape_cb_plain(void);

#cgo noescape shape_add_fast
#cgo nocallback shape_add_fast
#cgo noescape shape_u64_fast
#cgo nocallback shape_u64_fast
#cgo noescape shape_f32_fast
#cgo nocallback shape_f32_fast
#cgo noescape shape_f64_fast
#cgo nocallback shape_f64_fast
#cgo noescape shape_bool_fast
#cgo nocallback shape_bool_fast
#cgo noescape shape_nop_fast
#cgo nocallback shape_nop_fast
#cgo noescape shape_void_fast
#cgo nocallback shape_void_fast
#cgo noescape shape_read_fast
#cgo nocallback shape_read_fast
#cgo noescape shape_ppre_fast
#cgo nocallback shape_ppre_fast
#cgo noescape shape_sum_fast
#cgo nocallback shape_sum_fast
#cgo noescape shape_point_fast
#cgo nocallback shape_point_fast
#cgo noescape shape_color_fast
#cgo nocallback shape_color_fast
#cgo noescape shape_apply_fast
#cgo nocallback shape_apply_fast
#cgo noescape shape_errno_fast
#cgo nocallback shape_errno_fast
#cgo noescape shape_inline_fast
#cgo nocallback shape_inline_fast
#cgo noescape cgoCheckPointer
#cgo nocallback cgoCheckPointer
#cgo noescape cgocall
#cgo nocallback cgocall
#cgo noescape shape_strlen_fast
#cgo nocallback shape_strlen_fast
#cgo noescape shape_vp_fast
#cgo nocallback shape_vp_fast
#cgo noescape shape_cb_fast
#cgo nocallback shape_cb_fast
#cgo nocallback shape_cb_plain
*/
import "C"

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

//export GoCallback
func GoCallback(v C.int) C.int { return v + 1 }

func fail(msg string) {
	fmt.Println("FAIL:", msg)
	os.Exit(1)
}

func callOnce(fn func() int32) string {
	msg := ""
	func() {
		defer func() {
			if r := recover(); r != nil {
				msg, _ = r.(string)
			}
		}()
		fn()
		msg = "returned"
	}()
	return msg
}
func checkPanicsTwice(name string, fn func() int32, secondWant string) {
	first := callOnce(fn)
	if first == "returned" {
		fail(name + " survived callback")
	}
	if !strings.Contains(first, "marked with #cgo nocallback") {
		fail(name + " first panic: " + first)
	}
	second := callOnce(fn)
	if !strings.Contains(second, secondWant) {
		fail(name + " second panic: " + second)
	}
}
func main() {
	if int(C.shape_add_fast(40, 2)) != 42 {
		fail("add fast")
	}
	if int(C.shape_add_slow(40, 2)) != 42 {
		fail("add slow")
	}
	if uint64(C.shape_u64_fast(21)) != 43 {
		fail("u64 fast")
	}
	if uint64(C.shape_u64_slow(21)) != 43 {
		fail("u64 slow")
	}
	if float32(C.shape_f32_fast(2.5)) != 4.0 {
		fail("f32 fast")
	}
	if float32(C.shape_f32_slow(2.5)) != 4.0 {
		fail("f32 slow")
	}
	if float64(C.shape_f64_fast(21)) != 42.0 {
		fail("f64 fast")
	}
	if float64(C.shape_f64_slow(21)) != 42.0 {
		fail("f64 slow")
	}
	if bool(C.shape_bool_fast(true)) != false {
		fail("bool fast")
	}
	if bool(C.shape_bool_slow(true)) != false {
		fail("bool slow")
	}
	C.shape_nop_fast()
	C.shape_nop_slow()
	v := C.int(-99)
	if int(C.shape_read_fast(&v)) != -99 {
		fail("read fast")
	}
	if int(C.shape_read_slow(&v)) != -99 {
		fail("read slow")
	}
	mem := C.malloc(4)
	*(*C.int)(mem) = 40
	pmem := C.malloc(8)
	*(*unsafe.Pointer)(pmem) = mem
	if int(C.shape_ppre_fast((**C.int)(pmem))) != 40 {
		fail("ppre fast")
	}
	if int(C.shape_ppre_slow((**C.int)(pmem))) != 40 {
		fail("ppre slow")
	}
	C.free(mem)
	C.free(pmem)
	if int(C.shape_apply_fast(C.shape_fn_t(C.shape_double), 21)) != 42 {
		fail("apply fast")
	}
	if int(C.shape_apply_slow(C.shape_fn_t(C.shape_double), 21)) != 42 {
		fail("apply slow")
	}
	buf := []byte{10, 20, 30}
	if int(C.shape_sum_fast((*C.uchar)(unsafe.Pointer(&buf[0])), 3)) != 60 {
		fail("sum fast")
	}
	if int(C.shape_sum_slow((*C.uchar)(unsafe.Pointer(&buf[0])), 3)) != 60 {
		fail("sum slow")
	}
	pt := C.struct_ShapePoint{x: 3, y: 4}
	if int(C.shape_point_fast(pt)) != 7 {
		fail("point fast")
	}
	if int(C.shape_point_slow(pt)) != 7 {
		fail("point slow")
	}
	if int(C.shape_color_fast(C.SHAPE_GREEN)) != 2 {
		fail("color fast")
	}
	if int(C.shape_color_slow(C.SHAPE_GREEN)) != 2 {
		fail("color slow")
	}
	r1, err1 := C.shape_errno_fast(-1)
	if int(r1) != -1 || err1 == nil {
		fail("errno fast error")
	}
	r2, err2 := C.shape_errno_fast(21)
	if int(r2) != 42 || err2 != nil {
		fail("errno fast stale")
	}
	r3, err3 := C.shape_errno_slow(-1)
	if int(r3) != -1 || err3 == nil {
		fail("errno slow error")
	}
	r4, err4 := C.shape_errno_slow(21)
	if int(r4) != 42 || err4 != nil {
		fail("errno slow stale")
	}
	var wg sync.WaitGroup
	bad := make(chan string, 64)
	for i := range 8 {
		wg.Go(func() {
			for j := range 100 {
				if (j+i)%2 == 0 {
					rr, ee := C.shape_errno_fast(-1)
					if int(rr) != -1 || ee == nil {
						bad <- "concurrent fast error"
						return
					}
				} else {
					rr, ee := C.shape_errno_fast(7)
					if int(rr) != 14 || ee != nil {
						bad <- "concurrent fast ok"
						return
					}
				}
			}
		})
	}
	wg.Wait()
	select {
	case m := <-bad:
		fail(m)
	default:
	}
	if int(C.shape_inline_fast(7)) != 22 {
		fail("inline fast")
	}
	if int(C.shape_inline_slow(7)) != 22 {
		fail("inline slow")
	}
	if int(C.cgoCheckPointer(41)) != 42 {
		fail("collide checkpointer")
	}
	if int(C.cgocall(40)) != 42 {
		fail("collide cgocall")
	}
	cs := C.CString("hello")
	if int(C.shape_strlen_fast(cs)) != 5 {
		fail("strlen fast")
	}
	if int(C.shape_strlen_slow(cs)) != 5 {
		fail("strlen slow")
	}
	C.free(unsafe.Pointer(cs))
	iv := C.int(99)
	if int(C.shape_vp_fast(unsafe.Pointer(&iv))) != 99 {
		fail("vp fast")
	}
	if int(C.shape_vp_slow(unsafe.Pointer(&iv))) != 99 {
		fail("vp slow")
	}
	wv := C.int(0)
	C.shape_void_fast(&wv)
	if int(wv) != 42 {
		fail("void fast")
	}
	C.shape_void_slow(&wv)
	if int(wv) != 42 {
		fail("void slow")
	}
	var ta C.int = 42
	if int(ta) != 42 {
		fail("ctype alias")
	}
	mode := "fast"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	if mode == "plain" {
		checkPanicsTwice("cb plain", func() int32 { return int32(C.shape_cb_plain()) }, "unexpected setting cgoNoCallback")
	} else {
		checkPanicsTwice("cb fast", func() int32 { return int32(C.shape_cb_fast()) }, "marked with #cgo nocallback")
	}
	if int(C.shape_add_slow(20, 22)) != 42 {
		fail("add after panic")
	}
	fmt.Println("SHAPES OK")
}
