// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

/*
static void cgo_span_consume(void *p) { (void)p; }
static void cgo_span_bytes(const unsigned char *p) { (void)p; }
*/
import "C"

import "unsafe"

func callCgoSpan(p unsafe.Pointer) {
	C.cgo_span_consume(p)
}

func callCgoSpanBytes(p *byte) {
	C.cgo_span_bytes((*C.uchar)(unsafe.Pointer(p)))
}
