// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

// Syms holds known symbols.
var Syms struct {
	AssertE2I         *obj.LSym
	AssertE2I2        *obj.LSym
	AssertI2I         *obj.LSym
	AssertI2I2        *obj.LSym
	Asanread          *obj.LSym
	Asanwrite         *obj.LSym
	CheckPtrAlignment *obj.LSym
	Deferproc         *obj.LSym
	DeferprocStack    *obj.LSym
	Deferreturn       *obj.LSym
	Duffcopy          *obj.LSym
	Duffzero          *obj.LSym
	GCWriteBarrier    *obj.LSym
	Goschedguarded    *obj.LSym
	Growslice         *obj.LSym
	Memmove           *obj.LSym
	Msanread          *obj.LSym
	Msanwrite         *obj.LSym
	Msanmove          *obj.LSym
	Newobject         *obj.LSym
	Newproc           *obj.LSym
	Panicdivide       *obj.LSym
	Panicshift        *obj.LSym
	PanicdottypeE     *obj.LSym
	PanicdottypeI     *obj.LSym
	Panicnildottype   *obj.LSym
	Panicoverflow     *obj.LSym
	Raceread          *obj.LSym
	Racereadrange     *obj.LSym
	Racewrite         *obj.LSym
	Racewriterange    *obj.LSym
	// Wasm
	SigPanic        *obj.LSym
	Staticuint64s   *obj.LSym
	Typedmemclr     *obj.LSym
	Typedmemmove    *obj.LSym
	Udiv            *obj.LSym
	WriteBarrier    *obj.LSym
	Zerobase        *obj.LSym
	ARM64HasATOMICS *obj.LSym
	ARMHasVFPv4     *obj.LSym
	X86HasFMA       *obj.LSym
	X86HasPOPCNT    *obj.LSym
	X86HasSSE41     *obj.LSym
	// Wasm
	WasmDiv *obj.LSym
	// Wasm
	WasmMove *obj.LSym
	// Wasm
	WasmZero *obj.LSym
	// Wasm
	WasmTruncS *obj.LSym
	// Wasm
	WasmTruncU *obj.LSym
}

// Pkgs holds known packages.
var Pkgs struct {
	Go      *types.Pkg
	Itab    *types.Pkg
	Runtime *types.Pkg
}
