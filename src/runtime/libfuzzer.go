// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build libfuzzer

package runtime

import _ "unsafe" // for go:linkname

func libfuzzerCall(fn *byte, arg0, arg1 uintptr)

func libfuzzerTraceCmp1(arg0, arg1 uint8) {
	libfuzzerCall(&__sanitizer_cov_trace_cmp1, uintptr(arg0), uintptr(arg1))
}

func libfuzzerTraceCmp2(arg0, arg1 uint16) {
	libfuzzerCall(&__sanitizer_cov_trace_cmp2, uintptr(arg0), uintptr(arg1))
}

func libfuzzerTraceCmp4(arg0, arg1 uint32) {
	libfuzzerCall(&__sanitizer_cov_trace_cmp4, uintptr(arg0), uintptr(arg1))
}

func libfuzzerTraceCmp8(arg0, arg1 uint64) {
	libfuzzerCall(&__sanitizer_cov_trace_cmp8, uintptr(arg0), uintptr(arg1))
}

func libfuzzerTraceConstCmp1(arg0, arg1 uint8) {
	libfuzzerCall(&__sanitizer_cov_trace_const_cmp1, uintptr(arg0), uintptr(arg1))
}

func libfuzzerTraceConstCmp2(arg0, arg1 uint16) {
	libfuzzerCall(&__sanitizer_cov_trace_const_cmp2, uintptr(arg0), uintptr(arg1))
}

func libfuzzerTraceConstCmp4(arg0, arg1 uint32) {
	libfuzzerCall(&__sanitizer_cov_trace_const_cmp4, uintptr(arg0), uintptr(arg1))
}

func libfuzzerTraceConstCmp8(arg0, arg1 uint64) {
	libfuzzerCall(&__sanitizer_cov_trace_const_cmp8, uintptr(arg0), uintptr(arg1))
}

//go:linkname __sanitizer_cov_trace_cmp1 __sanitizer_cov_trace_cmp1
//go:cgo_import_static __sanitizer_cov_trace_cmp1
var __sanitizer_cov_trace_cmp1 byte

//go:linkname __sanitizer_cov_trace_cmp2 __sanitizer_cov_trace_cmp2
//go:cgo_import_static __sanitizer_cov_trace_cmp2
var __sanitizer_cov_trace_cmp2 byte

//go:linkname __sanitizer_cov_trace_cmp4 __sanitizer_cov_trace_cmp4
//go:cgo_import_static __sanitizer_cov_trace_cmp4
var __sanitizer_cov_trace_cmp4 byte

//go:linkname __sanitizer_cov_trace_cmp8 __sanitizer_cov_trace_cmp8
//go:cgo_import_static __sanitizer_cov_trace_cmp8
var __sanitizer_cov_trace_cmp8 byte

//go:linkname __sanitizer_cov_trace_const_cmp1 __sanitizer_cov_trace_const_cmp1
//go:cgo_import_static __sanitizer_cov_trace_const_cmp1
var __sanitizer_cov_trace_const_cmp1 byte

//go:linkname __sanitizer_cov_trace_const_cmp2 __sanitizer_cov_trace_const_cmp2
//go:cgo_import_static __sanitizer_cov_trace_const_cmp2
var __sanitizer_cov_trace_const_cmp2 byte

//go:linkname __sanitizer_cov_trace_const_cmp4 __sanitizer_cov_trace_const_cmp4
//go:cgo_import_static __sanitizer_cov_trace_const_cmp4
var __sanitizer_cov_trace_const_cmp4 byte

//go:linkname __sanitizer_cov_trace_const_cmp8 __sanitizer_cov_trace_const_cmp8
//go:cgo_import_static __sanitizer_cov_trace_const_cmp8
var __sanitizer_cov_trace_const_cmp8 byte
