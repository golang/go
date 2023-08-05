// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !libfuzzer

package fuzz

import _ "unsafe" // for go:linkname

//go:linkname libfuzzerTraceCmp1 runtime.libfuzzerTraceCmp1
//go:linkname libfuzzerTraceCmp2 runtime.libfuzzerTraceCmp2
//go:linkname libfuzzerTraceCmp4 runtime.libfuzzerTraceCmp4
//go:linkname libfuzzerTraceCmp8 runtime.libfuzzerTraceCmp8

//go:linkname libfuzzerTraceConstCmp1 runtime.libfuzzerTraceConstCmp1
//go:linkname libfuzzerTraceConstCmp2 runtime.libfuzzerTraceConstCmp2
//go:linkname libfuzzerTraceConstCmp4 runtime.libfuzzerTraceConstCmp4
//go:linkname libfuzzerTraceConstCmp8 runtime.libfuzzerTraceConstCmp8

//go:linkname libfuzzerHookStrCmp runtime.libfuzzerHookStrCmp
//go:linkname libfuzzerHookEqualFold runtime.libfuzzerHookEqualFold

func libfuzzerTraceCmp1(arg0, arg1 uint8, fakePC uint)  {}
func libfuzzerTraceCmp2(arg0, arg1 uint16, fakePC uint) {}
func libfuzzerTraceCmp4(arg0, arg1 uint32, fakePC uint) {}
func libfuzzerTraceCmp8(arg0, arg1 uint64, fakePC uint) {}

func libfuzzerTraceConstCmp1(arg0, arg1 uint8, fakePC uint)  {}
func libfuzzerTraceConstCmp2(arg0, arg1 uint16, fakePC uint) {}
func libfuzzerTraceConstCmp4(arg0, arg1 uint32, fakePC uint) {}
func libfuzzerTraceConstCmp8(arg0, arg1 uint64, fakePC uint) {}

func libfuzzerHookStrCmp(arg0, arg1 string, fakePC uint)    {}
func libfuzzerHookEqualFold(arg0, arg1 string, fakePC uint) {}
