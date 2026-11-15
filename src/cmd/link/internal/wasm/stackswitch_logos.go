// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"cmd/internal/obj/wasm"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
)

const sectionTag = 13

func init() {
	wasmFuncTypes["wasm_cont_run"] = &wasmFuncType{Params: []byte{}}
	wasmFuncTypes["wasm_cont_entry"] = &wasmFuncType{Params: []byte{}}
}

var stackSwitch bool

func logosStackSwitchTypes(ldr *loader.Loader, types []*wasmFuncType) []*wasmFuncType {
	s := ldr.Lookup("wasm_cont_entry", 0)
	stackSwitch = s != 0 && ldr.AttrReachable(s)
	if !stackSwitch {
		return types
	}
	if len(types) != wasm.ContFuncType {
		panic("wasm: stack-switching types are not at their fixed indices")
	}
	return append(types, &wasmFuncType{Params: []byte{}, Results: []byte{}}, &wasmFuncType{cont: true})
}

func logosWriteContType(ctxt *ld.Link) {
	ctxt.Out.WriteByte(0x5d)
	writeUleb128(ctxt.Out, wasm.ContFuncType)
}

func logosContTables() uint64 {
	if stackSwitch {
		return 1
	}
	return 0
}

func logosWriteContTable(ctxt *ld.Link) {
	if !stackSwitch {
		return
	}
	ctxt.Out.WriteByte(0x63)
	writeUleb128(ctxt.Out, wasm.ContType)
	ctxt.Out.WriteByte(0x00)
	writeUleb128(ctxt.Out, 0)
}

func logosWriteTagSec(ctxt *ld.Link) {
	if !stackSwitch {
		return
	}
	sizeOffset := writeSecHeader(ctxt, sectionTag)
	writeUleb128(ctxt.Out, 1)
	ctxt.Out.WriteByte(0x00)
	writeUleb128(ctxt.Out, wasm.ContFuncType)
	writeSecSize(ctxt, sizeOffset)
}
