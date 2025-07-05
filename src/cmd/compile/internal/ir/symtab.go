// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

// Syms holds known symbols.
var Syms symsStruct

type symsStruct struct {
	AssertE2I                 *obj.LSym
	AssertE2I2                *obj.LSym
	Asanread                  *obj.LSym
	Asanwrite                 *obj.LSym
	CgoCheckMemmove           *obj.LSym
	CgoCheckPtrWrite          *obj.LSym
	CheckPtrAlignment         *obj.LSym
	Deferproc                 *obj.LSym
	Deferprocat               *obj.LSym
	DeferprocStack            *obj.LSym
	Deferreturn               *obj.LSym
	Duffcopy                  *obj.LSym
	Duffzero                  *obj.LSym
	GCWriteBarrier            [8]*obj.LSym
	Goschedguarded            *obj.LSym
	Growslice                 *obj.LSym
	GrowsliceBuf              *obj.LSym
	MoveSlice                 *obj.LSym
	MoveSliceNoScan           *obj.LSym
	MoveSliceNoCap            *obj.LSym
	MoveSliceNoCapNoScan      *obj.LSym
	InterfaceSwitch           *obj.LSym
	MallocGC                  *obj.LSym
	MallocGCSmallNoScan       [27]*obj.LSym
	MallocGCSmallScanNoHeader [27]*obj.LSym
	MallocGCTiny              [16]*obj.LSym
	Memmove                   *obj.LSym
	Memequal                  *obj.LSym
	Msanread                  *obj.LSym
	Msanwrite                 *obj.LSym
	Msanmove                  *obj.LSym
	Newobject                 *obj.LSym
	Newproc                   *obj.LSym
	PanicBounds               *obj.LSym
	PanicExtend               *obj.LSym
	Panicdivide               *obj.LSym
	Panicshift                *obj.LSym
	PanicdottypeE             *obj.LSym
	PanicdottypeI             *obj.LSym
	Panicnildottype           *obj.LSym
	Panicoverflow             *obj.LSym
	PanicSimdImm              *obj.LSym
	Racefuncenter             *obj.LSym
	Racefuncexit              *obj.LSym
	Raceread                  *obj.LSym
	Racereadrange             *obj.LSym
	Racewrite                 *obj.LSym
	Racewriterange            *obj.LSym
	TypeAssert                *obj.LSym
	WBZero                    *obj.LSym
	WBMove                    *obj.LSym
	// Wasm
	SigPanic         *obj.LSym
	Staticuint64s    *obj.LSym
	Typedmemmove     *obj.LSym
	Udiv             *obj.LSym
	WriteBarrier     *obj.LSym
	Zerobase         *obj.LSym
	ZeroVal          *obj.LSym
	ARM64HasATOMICS  *obj.LSym
	ARMHasVFPv4      *obj.LSym
	Loong64HasLAMCAS *obj.LSym
	Loong64HasLAM_BH *obj.LSym
	Loong64HasLSX    *obj.LSym
	RISCV64HasZbb    *obj.LSym
	X86HasAVX        *obj.LSym
	X86HasFMA        *obj.LSym
	X86HasPOPCNT     *obj.LSym
	X86HasSSE41      *obj.LSym
	// Wasm
	WasmDiv *obj.LSym
	// Wasm
	WasmTruncS *obj.LSym
	// Wasm
	WasmTruncU *obj.LSym
}

// Pkgs holds known packages.
var Pkgs struct {
	Go           *types.Pkg
	Itab         *types.Pkg
	Runtime      *types.Pkg
	InternalMaps *types.Pkg
	Coverage     *types.Pkg
}
