// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/goobj2"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"container/heap"
	"fmt"
	"unicode"
)

var _ = fmt.Print

type workQueue []loader.Sym

// Implement container/heap.Interface.
func (q *workQueue) Len() int           { return len(*q) }
func (q *workQueue) Less(i, j int) bool { return (*q)[i] < (*q)[j] }
func (q *workQueue) Swap(i, j int)      { (*q)[i], (*q)[j] = (*q)[j], (*q)[i] }
func (q *workQueue) Push(i interface{}) { *q = append(*q, i.(loader.Sym)) }
func (q *workQueue) Pop() interface{}   { i := (*q)[len(*q)-1]; *q = (*q)[:len(*q)-1]; return i }

// Functions for deadcode pass to use.
// Deadcode pass should call push/pop, not Push/Pop.
func (q *workQueue) push(i loader.Sym) { heap.Push(q, i) }
func (q *workQueue) pop() loader.Sym   { return heap.Pop(q).(loader.Sym) }
func (q *workQueue) empty() bool       { return len(*q) == 0 }

type deadcodePass struct {
	ctxt *Link
	ldr  *loader.Loader
	wq   workQueue

	ifaceMethod     map[methodsig]bool // methods declared in reached interfaces
	markableMethods []methodref        // methods of reached types
	reflectSeen     bool               // whether we have seen a reflect method call
}

func (d *deadcodePass) init() {
	d.ldr.InitReachable()
	d.ifaceMethod = make(map[methodsig]bool)
	if objabi.Fieldtrack_enabled != 0 {
		d.ldr.Reachparent = make([]loader.Sym, d.ldr.NSym())
	}
	heap.Init(&d.wq)

	if d.ctxt.BuildMode == BuildModeShared {
		// Mark all symbols defined in this library as reachable when
		// building a shared library.
		n := d.ldr.NDef()
		for i := 1; i < n; i++ {
			s := loader.Sym(i)
			d.mark(s, 0)
		}
		return
	}

	var names []string

	// In a normal binary, start at main.main and the init
	// functions and mark what is reachable from there.
	if d.ctxt.linkShared && (d.ctxt.BuildMode == BuildModeExe || d.ctxt.BuildMode == BuildModePIE) {
		names = append(names, "main.main", "main..inittask")
	} else {
		// The external linker refers main symbol directly.
		if d.ctxt.LinkMode == LinkExternal && (d.ctxt.BuildMode == BuildModeExe || d.ctxt.BuildMode == BuildModePIE) {
			if d.ctxt.HeadType == objabi.Hwindows && d.ctxt.Arch.Family == sys.I386 {
				*flagEntrySymbol = "_main"
			} else {
				*flagEntrySymbol = "main"
			}
		}
		names = append(names, *flagEntrySymbol)
		if d.ctxt.BuildMode == BuildModePlugin {
			names = append(names, objabi.PathToPrefix(*flagPluginPath)+"..inittask", objabi.PathToPrefix(*flagPluginPath)+".main", "go.plugin.tabs")

			// We don't keep the go.plugin.exports symbol,
			// but we do keep the symbols it refers to.
			exportsIdx := d.ldr.Lookup("go.plugin.exports", 0)
			if exportsIdx != 0 {
				relocs := d.ldr.Relocs(exportsIdx)
				for i := 0; i < relocs.Count(); i++ {
					d.mark(relocs.At2(i).Sym(), 0)
				}
			}
		}
	}

	dynexpMap := d.ctxt.cgo_export_dynamic
	if d.ctxt.LinkMode == LinkExternal {
		dynexpMap = d.ctxt.cgo_export_static
	}
	for exp := range dynexpMap {
		names = append(names, exp)
	}

	for _, name := range names {
		// Mark symbol as a data/ABI0 symbol.
		d.mark(d.ldr.Lookup(name, 0), 0)
		// Also mark any Go functions (internal ABI).
		d.mark(d.ldr.Lookup(name, sym.SymVerABIInternal), 0)
	}
}

func (d *deadcodePass) flood() {
	for !d.wq.empty() {
		symIdx := d.wq.pop()

		d.reflectSeen = d.reflectSeen || d.ldr.IsReflectMethod(symIdx)

		isgotype := d.ldr.IsGoType(symIdx)
		relocs := d.ldr.Relocs(symIdx)

		if isgotype {
			p := d.ldr.Data(symIdx)
			if len(p) != 0 && decodetypeKind(d.ctxt.Arch, p)&kindMask == kindInterface {
				for _, sig := range d.decodeIfaceMethods(d.ldr, d.ctxt.Arch, symIdx, &relocs) {
					if d.ctxt.Debugvlog > 1 {
						d.ctxt.Logf("reached iface method: %s\n", sig)
					}
					d.ifaceMethod[sig] = true
				}
			}
		}

		var methods []methodref
		for i := 0; i < relocs.Count(); i++ {
			r := relocs.At2(i)
			t := r.Type()
			if t == objabi.R_WEAKADDROFF {
				continue
			}
			if t == objabi.R_METHODOFF {
				if i+2 >= relocs.Count() {
					panic("expect three consecutive R_METHODOFF relocs")
				}
				methods = append(methods, methodref{src: symIdx, r: i})
				i += 2
				continue
			}
			if t == objabi.R_USETYPE {
				// type symbol used for DWARF. we need to load the symbol but it may not
				// be otherwise reachable in the program.
				// do nothing for now as we still load all type symbols.
				continue
			}
			d.mark(r.Sym(), symIdx)
		}
		naux := d.ldr.NAux(symIdx)
		for i := 0; i < naux; i++ {
			a := d.ldr.Aux2(symIdx, i)
			if a.Type() == goobj2.AuxGotype && !d.ctxt.linkShared {
				// A symbol being reachable doesn't imply we need its
				// type descriptor. Don't mark it.
				// TODO: when -linkshared, the GCProg generation code
				// seems to need it. I'm not sure why. I think it could
				// just reach to the type descriptor's data without
				// requiring to mark it reachable.
				continue
			}
			d.mark(a.Sym(), symIdx)
		}
		// Some host object symbols have an outer object, which acts like a
		// "carrier" symbol, or it holds all the symbols for a particular
		// section. We need to mark all "referenced" symbols from that carrier,
		// so we make sure we're pulling in all outer symbols, and their sub
		// symbols. This is not ideal, and these carrier/section symbols could
		// be removed.
		if d.ldr.IsExternal(symIdx) {
			d.mark(d.ldr.OuterSym(symIdx), symIdx)
			d.mark(d.ldr.SubSym(symIdx), symIdx)
		}

		if len(methods) != 0 {
			if !isgotype {
				panic("method found on non-type symbol")
			}
			// Decode runtime type information for type methods
			// to help work out which methods can be called
			// dynamically via interfaces.
			methodsigs := d.decodetypeMethods(d.ldr, d.ctxt.Arch, symIdx, &relocs)
			if len(methods) != len(methodsigs) {
				panic(fmt.Sprintf("%q has %d method relocations for %d methods", d.ldr.SymName(symIdx), len(methods), len(methodsigs)))
			}
			for i, m := range methodsigs {
				methods[i].m = m
			}
			d.markableMethods = append(d.markableMethods, methods...)
		}
	}
}

func (d *deadcodePass) mark(symIdx, parent loader.Sym) {
	if symIdx != 0 && !d.ldr.AttrReachable(symIdx) {
		d.wq.push(symIdx)
		d.ldr.SetAttrReachable(symIdx, true)
		if objabi.Fieldtrack_enabled != 0 {
			d.ldr.Reachparent[symIdx] = parent
		}
		if *flagDumpDep {
			to := d.ldr.SymName(symIdx)
			if to != "" {
				from := "_"
				if parent != 0 {
					from = d.ldr.SymName(parent)
				}
				fmt.Printf("%s -> %s\n", from, to)
			}
		}
	}
}

func (d *deadcodePass) markMethod(m methodref) {
	relocs := d.ldr.Relocs(m.src)
	d.mark(relocs.At2(m.r).Sym(), m.src)
	d.mark(relocs.At2(m.r+1).Sym(), m.src)
	d.mark(relocs.At2(m.r+2).Sym(), m.src)
}

// deadcode marks all reachable symbols.
//
// The basis of the dead code elimination is a flood fill of symbols,
// following their relocations, beginning at *flagEntrySymbol.
//
// This flood fill is wrapped in logic for pruning unused methods.
// All methods are mentioned by relocations on their receiver's *rtype.
// These relocations are specially defined as R_METHODOFF by the compiler
// so we can detect and manipulated them here.
//
// There are three ways a method of a reachable type can be invoked:
//
//	1. direct call
//	2. through a reachable interface type
//	3. reflect.Value.Method (or MethodByName), or reflect.Type.Method
//	   (or MethodByName)
//
// The first case is handled by the flood fill, a directly called method
// is marked as reachable.
//
// The second case is handled by decomposing all reachable interface
// types into method signatures. Each encountered method is compared
// against the interface method signatures, if it matches it is marked
// as reachable. This is extremely conservative, but easy and correct.
//
// The third case is handled by looking to see if any of:
//	- reflect.Value.Method or MethodByName is reachable
// 	- reflect.Type.Method or MethodByName is called (through the
// 	  REFLECTMETHOD attribute marked by the compiler).
// If any of these happen, all bets are off and all exported methods
// of reachable types are marked reachable.
//
// Any unreached text symbols are removed from ctxt.Textp.
func deadcode(ctxt *Link) {
	ldr := ctxt.loader
	d := deadcodePass{ctxt: ctxt, ldr: ldr}
	d.init()
	d.flood()

	methSym := ldr.Lookup("reflect.Value.Method", sym.SymVerABIInternal)
	methByNameSym := ldr.Lookup("reflect.Value.MethodByName", sym.SymVerABIInternal)

	if ctxt.DynlinkingGo() {
		// Exported methods may satisfy interfaces we don't know
		// about yet when dynamically linking.
		d.reflectSeen = true
	}

	for {
		// Methods might be called via reflection. Give up on
		// static analysis, mark all exported methods of
		// all reachable types as reachable.
		d.reflectSeen = d.reflectSeen || (methSym != 0 && ldr.AttrReachable(methSym)) || (methByNameSym != 0 && ldr.AttrReachable(methByNameSym))

		// Mark all methods that could satisfy a discovered
		// interface as reachable. We recheck old marked interfaces
		// as new types (with new methods) may have been discovered
		// in the last pass.
		rem := d.markableMethods[:0]
		for _, m := range d.markableMethods {
			if (d.reflectSeen && m.isExported()) || d.ifaceMethod[m.m] {
				d.markMethod(m)
			} else {
				rem = append(rem, m)
			}
		}
		d.markableMethods = rem

		if d.wq.empty() {
			// No new work was discovered. Done.
			break
		}
		d.flood()
	}

	n := ldr.NSym()

	if ctxt.BuildMode != BuildModeShared {
		// Keep a itablink if the symbol it points at is being kept.
		// (When BuildModeShared, always keep itablinks.)
		for i := 1; i < n; i++ {
			s := loader.Sym(i)
			if ldr.IsItabLink(s) {
				relocs := ldr.Relocs(s)
				if relocs.Count() > 0 && ldr.AttrReachable(relocs.At2(0).Sym()) {
					ldr.SetAttrReachable(s, true)
				}
			}
		}
	}
}

// methodref holds the relocations from a receiver type symbol to its
// method. There are three relocations, one for each of the fields in
// the reflect.method struct: mtyp, ifn, and tfn.
type methodref struct {
	m   methodsig
	src loader.Sym // receiver type symbol
	r   int        // the index of R_METHODOFF relocations
}

func (m methodref) isExported() bool {
	for _, r := range m.m {
		return unicode.IsUpper(r)
	}
	panic("methodref has no signature")
}

// decodeMethodSig2 decodes an array of method signature information.
// Each element of the array is size bytes. The first 4 bytes is a
// nameOff for the method name, and the next 4 bytes is a typeOff for
// the function type.
//
// Conveniently this is the layout of both runtime.method and runtime.imethod.
func (d *deadcodePass) decodeMethodSig(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, off, size, count int) []methodsig {
	var buf bytes.Buffer
	var methods []methodsig
	for i := 0; i < count; i++ {
		buf.WriteString(decodetypeName(ldr, symIdx, relocs, off))
		mtypSym := decodeRelocSym(ldr, symIdx, relocs, int32(off+4))
		// FIXME: add some sort of caching here, since we may see some of the
		// same symbols over time for param types.
		mrelocs := ldr.Relocs(mtypSym)
		mp := ldr.Data(mtypSym)

		buf.WriteRune('(')
		inCount := decodetypeFuncInCount(arch, mp)
		for i := 0; i < inCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			a := decodetypeFuncInType(ldr, arch, mtypSym, &mrelocs, i)
			buf.WriteString(ldr.SymName(a))
		}
		buf.WriteString(") (")
		outCount := decodetypeFuncOutCount(arch, mp)
		for i := 0; i < outCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			a := decodetypeFuncOutType(ldr, arch, mtypSym, &mrelocs, i)
			buf.WriteString(ldr.SymName(a))
		}
		buf.WriteRune(')')

		off += size
		methods = append(methods, methodsig(buf.String()))
		buf.Reset()
	}
	return methods
}

func (d *deadcodePass) decodeIfaceMethods(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs) []methodsig {
	p := ldr.Data(symIdx)
	if decodetypeKind(arch, p)&kindMask != kindInterface {
		panic(fmt.Sprintf("symbol %q is not an interface", ldr.SymName(symIdx)))
	}
	rel := decodeReloc(ldr, symIdx, relocs, int32(commonsize(arch)+arch.PtrSize))
	s := rel.Sym()
	if s == 0 {
		return nil
	}
	if s != symIdx {
		panic(fmt.Sprintf("imethod slice pointer in %q leads to a different symbol", ldr.SymName(symIdx)))
	}
	off := int(rel.Add()) // array of reflect.imethod values
	numMethods := int(decodetypeIfaceMethodCount(arch, p))
	sizeofIMethod := 4 + 4
	return d.decodeMethodSig(ldr, arch, symIdx, relocs, off, sizeofIMethod, numMethods)
}

func (d *deadcodePass) decodetypeMethods(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs) []methodsig {
	p := ldr.Data(symIdx)
	if !decodetypeHasUncommon(arch, p) {
		panic(fmt.Sprintf("no methods on %q", ldr.SymName(symIdx)))
	}
	off := commonsize(arch) // reflect.rtype
	switch decodetypeKind(arch, p) & kindMask {
	case kindStruct: // reflect.structType
		off += 4 * arch.PtrSize
	case kindPtr: // reflect.ptrType
		off += arch.PtrSize
	case kindFunc: // reflect.funcType
		off += arch.PtrSize // 4 bytes, pointer aligned
	case kindSlice: // reflect.sliceType
		off += arch.PtrSize
	case kindArray: // reflect.arrayType
		off += 3 * arch.PtrSize
	case kindChan: // reflect.chanType
		off += 2 * arch.PtrSize
	case kindMap: // reflect.mapType
		off += 4*arch.PtrSize + 8
	case kindInterface: // reflect.interfaceType
		off += 3 * arch.PtrSize
	default:
		// just Sizeof(rtype)
	}

	mcount := int(decodeInuxi(arch, p[off+4:], 2))
	moff := int(decodeInuxi(arch, p[off+4+2+2:], 4))
	off += moff                // offset to array of reflect.method values
	const sizeofMethod = 4 * 4 // sizeof reflect.method in program
	return d.decodeMethodSig(ldr, arch, symIdx, relocs, off, sizeofMethod, mcount)
}
