// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/staticdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
)

// useNewABIWrapGen returns TRUE if the compiler should generate an
// ABI wrapper for the function 'f'.
func useABIWrapGen(f *ir.Func) bool {
	if !base.Flag.ABIWrap {
		return false
	}

	// Support limit option for bisecting.
	if base.Flag.ABIWrapLimit == 1 {
		return false
	}
	if base.Flag.ABIWrapLimit < 1 {
		return true
	}
	base.Flag.ABIWrapLimit--
	if base.Debug.ABIWrap != 0 && base.Flag.ABIWrapLimit == 1 {
		fmt.Fprintf(os.Stderr, "=-= limit reached after new wrapper for %s\n",
			f.LSym.Name)
	}

	return true
}

// symabiDefs and symabiRefs record the defined and referenced ABIs of
// symbols required by non-Go code. These are keyed by link symbol
// name, where the local package prefix is always `"".`
var symabiDefs, symabiRefs map[string]obj.ABI

func CgoSymABIs() {
	// The linker expects an ABI0 wrapper for all cgo-exported
	// functions.
	for _, prag := range typecheck.Target.CgoPragmas {
		switch prag[0] {
		case "cgo_export_static", "cgo_export_dynamic":
			if symabiRefs == nil {
				symabiRefs = make(map[string]obj.ABI)
			}
			symabiRefs[prag[1]] = obj.ABI0
		}
	}
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
func ReadSymABIs(file, myimportpath string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-symabis: %v", err)
	}

	symabiDefs = make(map[string]obj.ABI)
	symabiRefs = make(map[string]obj.ABI)

	localPrefix := ""
	if myimportpath != "" {
		// Symbols in this package may be written either as
		// "".X or with the package's import path already in
		// the symbol.
		localPrefix = objabi.PathToPrefix(myimportpath) + "."
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

			// If the symbol is already prefixed with
			// myimportpath, rewrite it to start with ""
			// so it matches the compiler's internal
			// symbol names.
			if localPrefix != "" && strings.HasPrefix(sym, localPrefix) {
				sym = `"".` + sym[len(localPrefix):]
			}

			// Record for later.
			if parts[0] == "def" {
				symabiDefs[sym] = abi
			} else {
				symabiRefs[sym] = abi
			}
		default:
			log.Fatalf(`%s:%d: invalid symabi type "%s"`, file, lineNum, parts[0])
		}
	}
}

// InitLSym defines f's obj.LSym and initializes it based on the
// properties of f. This includes setting the symbol flags and ABI and
// creating and initializing related DWARF symbols.
//
// InitLSym must be called exactly once per function and must be
// called for both functions with bodies and functions without bodies.
// For body-less functions, we only create the LSym; for functions
// with bodies call a helper to setup up / populate the LSym.
func InitLSym(f *ir.Func, hasBody bool) {
	// FIXME: for new-style ABI wrappers, we set up the lsym at the
	// point the wrapper is created.
	if f.LSym != nil && base.Flag.ABIWrap {
		return
	}
	staticdata.NeedFuncSym(f.Sym())
	selectLSym(f, hasBody)
	if hasBody {
		setupTextLSym(f, 0)
	}
}

// selectLSym sets up the LSym for a given function, and
// makes calls to helpers to create ABI wrappers if needed.
func selectLSym(f *ir.Func, hasBody bool) {
	if f.LSym != nil {
		base.FatalfAt(f.Pos(), "InitLSym called twice on %v", f)
	}

	if nam := f.Nname; !ir.IsBlank(nam) {

		var wrapperABI obj.ABI
		needABIWrapper := false
		defABI, hasDefABI := symabiDefs[nam.Linksym().Name]
		if hasDefABI && defABI == obj.ABI0 {
			// Symbol is defined as ABI0. Create an
			// Internal -> ABI0 wrapper.
			f.LSym = nam.LinksymABI(obj.ABI0)
			needABIWrapper, wrapperABI = true, obj.ABIInternal
		} else {
			f.LSym = nam.Linksym()
			// No ABI override. Check that the symbol is
			// using the expected ABI.
			want := obj.ABIInternal
			if f.LSym.ABI() != want {
				base.Fatalf("function symbol %s has the wrong ABI %v, expected %v", f.LSym.Name, f.LSym.ABI(), want)
			}
		}
		if f.Pragma&ir.Systemstack != 0 {
			f.LSym.Set(obj.AttrCFunc, true)
		}

		isLinknameExported := nam.Sym().Linkname != "" && (hasBody || hasDefABI)
		if abi, ok := symabiRefs[f.LSym.Name]; (ok && abi == obj.ABI0) || isLinknameExported {
			// Either 1) this symbol is definitely
			// referenced as ABI0 from this package; or 2)
			// this symbol is defined in this package but
			// given a linkname, indicating that it may be
			// referenced from another package. Create an
			// ABI0 -> Internal wrapper so it can be
			// called as ABI0. In case 2, it's important
			// that we know it's defined in this package
			// since other packages may "pull" symbols
			// using linkname and we don't want to create
			// duplicate ABI wrappers.
			if f.LSym.ABI() != obj.ABI0 {
				needABIWrapper, wrapperABI = true, obj.ABI0
			}
		}

		if needABIWrapper {
			if !useABIWrapGen(f) {
				// Fallback: use alias instead. FIXME.

				// These LSyms have the same name as the
				// native function, so we create them directly
				// rather than looking them up. The uniqueness
				// of f.lsym ensures uniqueness of asym.
				asym := &obj.LSym{
					Name: f.LSym.Name,
					Type: objabi.SABIALIAS,
					R:    []obj.Reloc{{Sym: f.LSym}}, // 0 size, so "informational"
				}
				asym.SetABI(wrapperABI)
				asym.Set(obj.AttrDuplicateOK, true)
				base.Ctxt.ABIAliases = append(base.Ctxt.ABIAliases, asym)
			} else {
				if base.Debug.ABIWrap != 0 {
					fmt.Fprintf(os.Stderr, "=-= %v to %v wrapper for %s.%s\n",
						wrapperABI, 1-wrapperABI, types.LocalPkg.Path, f.LSym.Name)
				}
				makeABIWrapper(f, wrapperABI)
			}
		}
	}
}

// makeABIWrapper creates a new function that wraps a cross-ABI call
// to "f".  The wrapper is marked as an ABIWRAPPER.
func makeABIWrapper(f *ir.Func, wrapperABI obj.ABI) {

	// Q: is this needed?
	savepos := base.Pos
	savedclcontext := typecheck.DeclContext
	savedcurfn := ir.CurFunc

	base.Pos = base.AutogeneratedPos
	typecheck.DeclContext = ir.PEXTERN

	// At the moment we don't support wrapping a method, we'd need machinery
	// below to handle the receiver. Panic if we see this scenario.
	ft := f.Nname.Ntype.Type()
	if ft.NumRecvs() != 0 {
		panic("makeABIWrapper support for wrapping methods not implemented")
	}

	// Manufacture a new func type to use for the wrapper.
	var noReceiver *ir.Field
	tfn := ir.NewFuncType(base.Pos,
		noReceiver,
		typecheck.NewFuncParams(ft.Params(), true),
		typecheck.NewFuncParams(ft.Results(), false))

	// Reuse f's types.Sym to create a new ODCLFUNC/function.
	fn := typecheck.DeclFunc(f.Nname.Sym(), tfn)
	fn.SetDupok(true)
	fn.SetWrapper(true) // ignore frame for panic+recover matching

	// Select LSYM now.
	asym := base.Ctxt.LookupABI(f.LSym.Name, wrapperABI)
	asym.Type = objabi.STEXT
	if fn.LSym != nil {
		panic("unexpected")
	}
	fn.LSym = asym

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
	setupTextLSym(fn, obj.NOSPLIT|obj.ABIWRAPPER)

	// Generate call. Use tail call if no params and no returns,
	// but a regular call otherwise.
	//
	// Note: ideally we would be using a tail call in cases where
	// there are params but no returns for ABI0->ABIInternal wrappers,
	// provided that all params fit into registers (e.g. we don't have
	// to allocate any stack space). Doing this will require some
	// extra work in typecheck/walk/ssa, might want to add a new node
	// OTAILCALL or something to this effect.
	tailcall := tfn.Type().NumResults() == 0 && tfn.Type().NumParams() == 0 && tfn.Type().NumRecvs() == 0
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
	if tailcall {
		tail = ir.NewTailCallStmt(base.Pos, f.Nname)
	} else {
		call := ir.NewCallExpr(base.Pos, ir.OCALL, f.Nname, nil)
		call.Args = ir.ParamNames(tfn.Type())
		call.IsDDD = tfn.Type().IsVariadic()
		tail = call
		if tfn.Type().NumResults() > 0 {
			n := ir.NewReturnStmt(base.Pos, nil)
			n.Results = []ir.Node{call}
			tail = n
		}
	}
	fn.Body.Append(tail)

	typecheck.FinishFuncBody()
	if base.Debug.DclStack != 0 {
		types.CheckDclstack()
	}

	typecheck.Func(fn)
	ir.CurFunc = fn
	typecheck.Stmts(fn.Body)

	escape.Batch([]*ir.Func{fn}, false)

	typecheck.Target.Decls = append(typecheck.Target.Decls, fn)

	// Restore previous context.
	base.Pos = savepos
	typecheck.DeclContext = savedclcontext
	ir.CurFunc = savedcurfn
}

// setupTextLsym initializes the LSym for a with-body text symbol.
func setupTextLSym(f *ir.Func, flag int) {
	if f.Dupok() {
		flag |= obj.DUPOK
	}
	if f.Wrapper() {
		flag |= obj.WRAPPER
	}
	if f.Needctxt() {
		flag |= obj.NEEDCTXT
	}
	if f.Pragma&ir.Nosplit != 0 {
		flag |= obj.NOSPLIT
	}
	if f.ReflectMethod() {
		flag |= obj.REFLECTMETHOD
	}

	// Clumsy but important.
	// See test/recover.go for test cases and src/reflect/value.go
	// for the actual functions being considered.
	if base.Ctxt.Pkgpath == "reflect" {
		switch f.Sym().Name {
		case "callReflect", "callMethod":
			flag |= obj.WRAPPER
		}
	}

	base.Ctxt.InitTextSym(f.LSym, flag)
}
