// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.pagetrace

package runtime

//go:systemstack
func pageTraceAlloc(pp *p, now int64, base, npages uintptr) {
}

//go:systemstack
func pageTraceFree(pp *p, now int64, base, npages uintptr) {
}

//go:systemstack
func pageTraceScav(pp *p, now int64, base, npages uintptr) {
}

type pageTraceBuf struct {
}

func initPageTrace(env string) {
}

func finishPageTrace() {
}
