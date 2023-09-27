// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgo

// These functions must be exported in order to perform
// longcall on cgo programs (cf gcc_aix_ppc64.c).
//
//go:cgo_export_static __cgo_topofstack
//go:cgo_export_static runtime.rt0_go
//go:cgo_export_static _rt0_ppc64_aix_lib
