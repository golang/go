// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"testing"
)

func TestLogosLoopStrideOnlyWasm(t *testing.T) {
	oldStride := base.Debug.LogosLoopStride
	base.Debug.LogosLoopStride = 4096
	defer func() { base.Debug.LogosLoopStride = oldStride }()
	oldPkg := base.Ctxt.Pkgpath
	base.Ctxt.Pkgpath = "main"
	defer func() { base.Ctxt.Pkgpath = oldPkg }()
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Ret("mem")))
	f := fun.f
	oldArch := f.Config.arch
	defer func() { f.Config.arch = oldArch }()
	f.Config.arch = "wasm"
	if got := logosLoopStride(f); got != 4096 {
		t.Fatalf("wasm logosLoopStride = %d, want 4096", got)
	}
	f.Config.arch = "amd64"
	if got := logosLoopStride(f); got != 0 {
		t.Fatalf("amd64 logosLoopStride = %d, want 0", got)
	}
	base.Debug.LogosLoopStride = 0
	f.Config.arch = "wasm"
	if got := logosLoopStride(f); got != 0 {
		t.Fatalf("wasm logosLoopStride with flag 0 = %d, want 0", got)
	}
}
