// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"fmt"
	"internal/buildcfg"
	"log"
	"os"
	"strings"

	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/wasm"

	rtabi "internal/abi"
)

// SymABIs records information provided by the assembler about symbol
// definition ABIs and reference ABIs.
type SymABIs struct {
	defs map[string]obj.ABI
	refs map[string]obj.ABISet
}

func NewSymABIs() *SymABIs {
	return &SymABIs{
		defs: make(map[string]obj.ABI),
		refs: make(map[string]obj.ABISet),
	}
}

// canonicalize returns the canonical name used for a linker symbol in
// s's maps. Symbols in this package may be written either as "".X or
// with the package's import path already in the symbol. This rewrites
// both to use the full path, which matches compiler-generated linker
// symbol names.
func (s *SymABIs) canonicalize(linksym string) string {
	if strings.HasPrefix(linksym, `"".`) {
		panic("non-canonical symbol name: " + linksym)
	}
	return linksym
}

// ReadSymABIs reads a symabis file that specifies definitions and
// references of text symbols by ABI.
//
// The symabis format is a set of lines, where each line is a sequence
// of whitespace-separated fields. The first field is a verb and is
// either "def" for defining a symbol ABI or "ref" for referencing a
// symbol using an ABI. For both "def" and "ref", the second field is
// the symbol name and the third field is the ABI name, as one of the
// named cmd/internal/obj.ABI constants.
func (s *SymABIs) ReadSymABIs(file string) {
	data, err := os.ReadFile(file)
	if err != nil {
		log.Fatalf("-symabis: %v", err)
	}

	for lineNum, line := range strings.Split(string(data), "\n") {
		lineNum++ // 1-based
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		switch parts[0] {
		case "def", "ref":
			// Parse line.
			if len(parts) != 3 {
				log.Fatalf(`%s:%d: invalid symabi: syntax is "%s sym abi"`, file, lineNum, parts[0])
			}
			sym, abistr := parts[1], parts[2]
			abi, valid := obj.ParseABI(abistr)
			if !valid {
				log.Fatalf(`%s:%d: invalid symabi: unknown abi "%s"`, file, lineNum, abistr)
			}

			sym = s.canonicalize(sym)

			// Record for later.
			if parts[0] == "def" {
				s.defs[sym] = abi
			} else {
				s.refs[sym] |= obj.ABISetOf(abi)
			}
		default:
			log.Fatalf(`%s:%d: invalid symabi type "%s"`, file, lineNum, parts[0])
		}
	}
}

// GenABIWrappers applies ABI information to Funcs and generates ABI
// wrapper functions where necessary.
func (s *SymABIs) GenABIWrappers() {
	// For cgo exported symbols, we tell the linker to export the
	// definition ABI to C. That also means that we don't want to
	// create ABI wrappers even if there's a linkname.
	//
	// TODO(austin): Maybe we want to create the ABI wrappers, but
	// ensure the linker exports the right ABI definition under
	// the unmangled name?
	cgoExports := make(map[string][]*[]string)
	for i, prag := range typecheck.Target.CgoPragmas {
		switch prag[0] {
		case "cgo_export_static", "cgo_export_dynamic":
			symName := s.canonicalize(prag[1])
			pprag := &typecheck.Target.CgoPragmas[i]
			cgoExports[symName] = append(cgoExports[symName], pprag)
		}
	}

	// Apply ABI defs and refs to Funcs and generate wrappers.
	//
	// This may generate new decls for the wrappers, but we
	// specifically *don't* want to visit those, lest we create
	// wrappers for wrappers.
	for _, fn := range typecheck.Target.Funcs {
		nam := fn.Nname
		if ir.IsBlank(nam) {
			continue
		}
		sym := nam.Sym()

		symName := sym.Linkname
		if symName == "" {
			symName = sym.Pkg.Prefix + "." + sym.Name
		}
		symName = s.canonicalize(symName)

		// Apply definitions.
		defABI, hasDefABI := s.defs[symName]
		if hasDefABI {
			if len(fn.Body) != 0 {
				base.ErrorfAt(fn.Pos(), 0, "%v defined in both Go and assembly", fn)
			}
			fn.ABI = defABI
		}

		if fn.Pragma&ir.CgoUnsafeArgs != 0 {
			// CgoUnsafeArgs indicates the function (or its callee) uses
			// offsets to dispatch arguments, which currently using ABI0
			// frame layout. Pin it to ABI0.
			fn.ABI = obj.ABI0
			// Propagate linkname attribute, which was set on the ABIInternal
			// symbol.
			if sym.Linksym().IsLinkname() {
				sym.LinksymABI(fn.ABI).Set(obj.AttrLinkname, true)
			}
		}

		// If cgo-exported, add the definition ABI to the cgo
		// pragmas.
		cgoExport := cgoExports[symName]
		for _, pprag := range cgoExport {
			// The export pragmas have the form:
			//
			//   cgo_export_* <local> [<remote>]
			//
			// If <remote> is omitted, it's the same as
			// <local>.
			//
			// Expand to
			//
			//   cgo_export_* <local> <remote> <ABI>
			if len(*pprag) == 2 {
				*pprag = append(*pprag, (*pprag)[1])
			}
			// Add the ABI argument.
			*pprag = append(*pprag, fn.ABI.String())
		}

		// Apply references.
		if abis, ok := s.refs[symName]; ok {
			fn.ABIRefs |= abis
		}
		// Assume all functions are referenced at least as
		// ABIInternal, since they may be referenced from
		// other packages.
		fn.ABIRefs.Set(obj.ABIInternal, true)

		// If a symbol is defined in this package (either in
		// Go or assembly) and given a linkname, it may be
		// referenced from another package, so make it
		// callable via any ABI. It's important that we know
		// it's defined in this package since other packages
		// may "pull" symbols using linkname and we don't want
		// to create duplicate ABI wrappers.
		//
		// However, if it's given a linkname for exporting to
		// C, then we don't make ABI wrappers because the cgo
		// tool wants the original definition.
		hasBody := len(fn.Body) != 0
		if sym.Linkname != "" && (hasBody || hasDefABI) && len(cgoExport) == 0 {
			fn.ABIRefs |= obj.ABISetCallable
		}

		// Double check that cgo-exported symbols don't get
		// any wrappers.
		if len(cgoExport) > 0 && fn.ABIRefs&^obj.ABISetOf(fn.ABI) != 0 {
			base.Fatalf("cgo exported function %v cannot have ABI wrappers", fn)
		}

		if !buildcfg.Experiment.RegabiWrappers {
			continue
		}

		forEachWrapperABI(fn, makeABIWrapper)
	}
}

func forEachWrapperABI(fn *ir.Func, cb func(fn *ir.Func, wrapperABI obj.ABI)) {
	need := fn.ABIRefs &^ obj.ABISetOf(fn.ABI)
	if need == 0 {
		return
	}

	for wrapperABI := obj.ABI(0); wrapperABI < obj.ABICount; wrapperABI++ {
		if !need.Get(wrapperABI) {
			continue
		}
		cb(fn, wrapperABI)
	}
}

// makeABIWrapper creates a new function that will be called with
// wrapperABI and calls "f" using f.ABI.
func makeABIWrapper(f *ir.Func, wrapperABI obj.ABI) {
	if base.Debug.ABIWrap != 0 {
		fmt.Fprintf(os.Stderr, "=-= %v to %v wrapper for %v\n", wrapperABI, f.ABI, f)
	}

	// Q: is this needed?
	savepos := base.Pos
	savedcurfn := ir.CurFunc

	pos := base.AutogeneratedPos
	base.Pos = pos

	// At the moment we don't support wrapping a method, we'd need machinery
	// below to handle the receiver. Panic if we see this scenario.
	ft := f.Nname.Type()
	if ft.NumRecvs() != 0 {
		base.ErrorfAt(f.Pos(), 0, "makeABIWrapper support for wrapping methods not implemented")
		return
	}

	// Reuse f's types.Sym to create a new ODCLFUNC/function.
	// TODO(mdempsky): Means we can't set sym.Def in Declfunc, ugh.
	fn := ir.NewFunc(pos, pos, f.Sym(), types.NewSignature(nil,
		typecheck.NewFuncParams(ft.Params()),
		typecheck.NewFuncParams(ft.Results())))
	fn.ABI = wrapperABI
	typecheck.DeclFunc(fn)

	fn.SetABIWrapper(true)
	fn.SetDupok(true)

	// ABI0-to-ABIInternal wrappers will be mainly loading params from
	// stack into registers (and/or storing stack locations back to
	// registers after the wrapped call); in most cases they won't
	// need to allocate stack space, so it should be OK to mark them
	// as NOSPLIT in these cases. In addition, my assumption is that
	// functions written in assembly are NOSPLIT in most (but not all)
	// cases. In the case of an ABIInternal target that has too many
	// parameters to fit into registers, the wrapper would need to
	// allocate stack space, but this seems like an unlikely scenario.
	// Hence: mark these wrappers NOSPLIT.
	//
	// ABIInternal-to-ABI0 wrappers on the other hand will be taking
	// things in registers and pushing them onto the stack prior to
	// the ABI0 call, meaning that they will always need to allocate
	// stack space. If the compiler marks them as NOSPLIT this seems
	// as though it could lead to situations where the linker's
	// nosplit-overflow analysis would trigger a link failure. On the
	// other hand if they not tagged NOSPLIT then this could cause
	// problems when building the runtime (since there may be calls to
	// asm routine in cases where it's not safe to grow the stack). In
	// most cases the wrapper would be (in effect) inlined, but are
	// there (perhaps) indirect calls from the runtime that could run
	// into trouble here.
	// FIXME: at the moment all.bash does not pass when I leave out
	// NOSPLIT for these wrappers, so all are currently tagged with NOSPLIT.
	fn.Pragma |= ir.Nosplit

	// Generate call. Use tail call if no params and no returns,
	// but a regular call otherwise.
	//
	// Note: ideally we would be using a tail call in cases where
	// there are params but no returns for ABI0->ABIInternal wrappers,
	// provided that all params fit into registers (e.g. we don't have
	// to allocate any stack space). Doing this will require some
	// extra work in typecheck/walk/ssa, might want to add a new node
	// OTAILCALL or something to this effect.
	tailcall := fn.Type().NumResults() == 0 && fn.Type().NumParams() == 0 && fn.Type().NumRecvs() == 0
	if base.Ctxt.Arch.Name == "ppc64le" && base.Ctxt.Flag_dynlink {
		// cannot tailcall on PPC64 with dynamic linking, as we need
		// to restore R2 after call.
		tailcall = false
	}
	if base.Ctxt.Arch.Name == "amd64" && wrapperABI == obj.ABIInternal {
		// cannot tailcall from ABIInternal to ABI0 on AMD64, as we need
		// to special registers (X15) when returning to ABIInternal.
		tailcall = false
	}

	var tail ir.Node
	call := ir.NewCallExpr(base.Pos, ir.OCALL, f.Nname, nil)
	call.Args = ir.ParamNames(fn.Type())
	call.IsDDD = fn.Type().IsVariadic()
	tail = call
	if tailcall {
		tail = ir.NewTailCallStmt(base.Pos, call)
	} else if fn.Type().NumResults() > 0 {
		n := ir.NewReturnStmt(base.Pos, nil)
		n.Results = []ir.Node{call}
		tail = n
	}
	fn.Body.Append(tail)

	typecheck.FinishFuncBody()

	ir.CurFunc = fn
	typecheck.Stmts(fn.Body)

	// Restore previous context.
	base.Pos = savepos
	ir.CurFunc = savedcurfn
}

// CreateWasmImportWrapper creates a wrapper for imported WASM functions to
// adapt them to the Go calling convention. The body for this function is
// generated in cmd/internal/obj/wasm/wasmobj.go
func CreateWasmImportWrapper(fn *ir.Func) bool {
	if fn.WasmImport == nil {
		return false
	}
	if buildcfg.GOARCH != "wasm" {
		base.FatalfAt(fn.Pos(), "CreateWasmImportWrapper call not supported on %s: func was %v", buildcfg.GOARCH, fn)
	}

	ir.InitLSym(fn, true)

	setupWasmImport(fn)

	pp := objw.NewProgs(fn, 0)
	defer pp.Free()
	pp.Text.To.Type = obj.TYPE_TEXTSIZE
	pp.Text.To.Val = int32(types.RoundUp(fn.Type().ArgWidth(), int64(types.RegSize)))
	// Wrapper functions never need their own stack frame
	pp.Text.To.Offset = 0
	pp.Flush()

	return true
}

func GenWasmExportWrapper(wrapped *ir.Func) {
	if wrapped.WasmExport == nil {
		return
	}
	if buildcfg.GOARCH != "wasm" {
		base.FatalfAt(wrapped.Pos(), "GenWasmExportWrapper call not supported on %s: func was %v", buildcfg.GOARCH, wrapped)
	}

	pos := base.AutogeneratedPos
	sym := &types.Sym{
		Name:     wrapped.WasmExport.Name,
		Linkname: wrapped.WasmExport.Name,
	}
	ft := wrapped.Nname.Type()
	fn := ir.NewFunc(pos, pos, sym, types.NewSignature(nil,
		typecheck.NewFuncParams(ft.Params()),
		typecheck.NewFuncParams(ft.Results())))
	fn.ABI = obj.ABI0 // actually wasm ABI
	// The wrapper function has a special calling convention that
	// morestack currently doesn't handle. For now we require that
	// the argument size fits in StackSmall, which we know we have
	// on stack, so we don't need to split stack.
	// cmd/internal/obj/wasm supports only 16 argument "registers"
	// anyway.
	if ft.ArgWidth() > rtabi.StackSmall {
		base.ErrorfAt(wrapped.Pos(), 0, "wasmexport function argument too large")
	}
	fn.Pragma |= ir.Nosplit

	ir.InitLSym(fn, true)

	setupWasmExport(fn, wrapped)

	pp := objw.NewProgs(fn, 0)
	defer pp.Free()
	// TEXT. Has a frame to pass args on stack to the Go function.
	pp.Text.To.Type = obj.TYPE_TEXTSIZE
	pp.Text.To.Val = int32(0)
	pp.Text.To.Offset = types.RoundUp(ft.ArgWidth(), int64(types.RegSize))
	// No locals. (Callee's args are covered in the callee's stackmap.)
	p := pp.Prog(obj.AFUNCDATA)
	p.From.SetConst(rtabi.FUNCDATA_LocalsPointerMaps)
	p.To.Type = obj.TYPE_MEM
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = base.Ctxt.Lookup("no_pointers_stackmap")
	pp.Flush()
	// Actual code geneneration is in cmd/internal/obj/wasm.
}

func paramsToWasmFields(f *ir.Func, pragma string, result *abi.ABIParamResultInfo, abiParams []abi.ABIParamAssignment) []obj.WasmField {
	wfs := make([]obj.WasmField, 0, len(abiParams))
	for _, p := range abiParams {
		t := p.Type
		var wt obj.WasmFieldType
		switch t.Kind() {
		case types.TINT32, types.TUINT32:
			wt = obj.WasmI32
		case types.TINT64, types.TUINT64:
			wt = obj.WasmI64
		case types.TFLOAT32:
			wt = obj.WasmF32
		case types.TFLOAT64:
			wt = obj.WasmF64
		case types.TUNSAFEPTR, types.TUINTPTR:
			wt = obj.WasmPtr
		case types.TBOOL:
			wt = obj.WasmBool
		case types.TSTRING:
			// Two parts, (ptr, len)
			wt = obj.WasmPtr
			wfs = append(wfs, obj.WasmField{Type: wt, Offset: p.FrameOffset(result)})
			wfs = append(wfs, obj.WasmField{Type: wt, Offset: p.FrameOffset(result) + int64(types.PtrSize)})
			continue
		case types.TPTR:
			if wasmElemTypeAllowed(t.Elem()) {
				wt = obj.WasmPtr
				break
			}
			fallthrough
		default:
			base.ErrorfAt(f.Pos(), 0, "%s: unsupported parameter type %s", pragma, t.String())
		}
		wfs = append(wfs, obj.WasmField{Type: wt, Offset: p.FrameOffset(result)})
	}
	return wfs
}

func resultsToWasmFields(f *ir.Func, pragma string, result *abi.ABIParamResultInfo, abiParams []abi.ABIParamAssignment) []obj.WasmField {
	if len(abiParams) > 1 {
		base.ErrorfAt(f.Pos(), 0, "%s: too many return values", pragma)
		return nil
	}
	wfs := make([]obj.WasmField, len(abiParams))
	for i, p := range abiParams {
		t := p.Type
		switch t.Kind() {
		case types.TINT32, types.TUINT32:
			wfs[i].Type = obj.WasmI32
		case types.TINT64, types.TUINT64:
			wfs[i].Type = obj.WasmI64
		case types.TFLOAT32:
			wfs[i].Type = obj.WasmF32
		case types.TFLOAT64:
			wfs[i].Type = obj.WasmF64
		case types.TUNSAFEPTR, types.TUINTPTR:
			wfs[i].Type = obj.WasmPtr
		case types.TBOOL:
			wfs[i].Type = obj.WasmBool
		case types.TPTR:
			if wasmElemTypeAllowed(t.Elem()) {
				wfs[i].Type = obj.WasmPtr
				break
			}
			fallthrough
		default:
			base.ErrorfAt(f.Pos(), 0, "%s: unsupported result type %s", pragma, t.String())
		}
		wfs[i].Offset = p.FrameOffset(result)
	}
	return wfs
}

// wasmElemTypeAllowed reports whether t is allowed to be passed in memory
// (as a pointer's element type, a field of it, etc.) between the Go wasm
// module and the host.
func wasmElemTypeAllowed(t *types.Type) bool {
	switch t.Kind() {
	case types.TINT8, types.TUINT8, types.TINT16, types.TUINT16,
		types.TINT32, types.TUINT32, types.TINT64, types.TUINT64,
		types.TFLOAT32, types.TFLOAT64, types.TBOOL:
		return true
	case types.TARRAY:
		return wasmElemTypeAllowed(t.Elem())
	case types.TSTRUCT:
		if len(t.Fields()) == 0 {
			return true
		}
		seenHostLayout := false
		for _, f := range t.Fields() {
			sym := f.Type.Sym()
			if sym != nil && sym.Name == "HostLayout" && sym.Pkg.Path == "structs" {
				seenHostLayout = true
				continue
			}
			if !wasmElemTypeAllowed(f.Type) {
				return false
			}
		}
		return seenHostLayout
	}
	// Pointer, and all pointerful types are not allowed, as pointers have
	// different width on the Go side and the host side. (It will be allowed
	// on GOARCH=wasm32.)
	return false
}

// setupWasmImport calculates the params and results in terms of WebAssembly values for the given function,
// and sets up the wasmimport metadata.
func setupWasmImport(f *ir.Func) {
	wi := obj.WasmImport{
		Module: f.WasmImport.Module,
		Name:   f.WasmImport.Name,
	}
	if wi.Module == wasm.GojsModule {
		// Functions that are imported from the "gojs" module use a special
		// ABI that just accepts the stack pointer.
		// Example:
		//
		// 	//go:wasmimport gojs add
		// 	func importedAdd(a, b uint) uint
		//
		// will roughly become
		//
		// 	(import "gojs" "add" (func (param i32)))
		wi.Params = []obj.WasmField{{Type: obj.WasmI32}}
	} else {
		// All other imported functions use the normal WASM ABI.
		// Example:
		//
		// 	//go:wasmimport a_module add
		// 	func importedAdd(a, b uint) uint
		//
		// will roughly become
		//
		// 	(import "a_module" "add" (func (param i32 i32) (result i32)))
		abiConfig := AbiForBodylessFuncStackMap(f)
		abiInfo := abiConfig.ABIAnalyzeFuncType(f.Type())
		wi.Params = paramsToWasmFields(f, "go:wasmimport", abiInfo, abiInfo.InParams())
		wi.Results = resultsToWasmFields(f, "go:wasmimport", abiInfo, abiInfo.OutParams())
	}
	f.LSym.Func().WasmImport = &wi
}

// setupWasmExport calculates the params and results in terms of WebAssembly values for the given function,
// and sets up the wasmexport metadata.
func setupWasmExport(f, wrapped *ir.Func) {
	we := obj.WasmExport{
		WrappedSym: wrapped.LSym,
	}
	abiConfig := AbiForBodylessFuncStackMap(wrapped)
	abiInfo := abiConfig.ABIAnalyzeFuncType(wrapped.Type())
	we.Params = paramsToWasmFields(wrapped, "go:wasmexport", abiInfo, abiInfo.InParams())
	we.Results = resultsToWasmFields(wrapped, "go:wasmexport", abiInfo, abiInfo.OutParams())
	f.LSym.Func().WasmExport = &we
}
