// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/dwarf"
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

type deadcodePass2 struct {
	ctxt *Link
	ldr  *loader.Loader
	wq   workQueue
	rtmp []loader.Reloc

	ifaceMethod     map[methodsig]bool // methods declared in reached interfaces
	markableMethods []methodref2       // methods of reached types
	reflectSeen     bool               // whether we have seen a reflect method call
}

func (d *deadcodePass2) init() {
	d.ldr.InitReachable()
	d.ifaceMethod = make(map[methodsig]bool)
	if d.ctxt.Reachparent != nil {
		d.ldr.Reachparent = make([]loader.Sym, d.ldr.NSym())
	}
	heap.Init(&d.wq)

	if d.ctxt.BuildMode == BuildModeShared {
		// Mark all symbols defined in this library as reachable when
		// building a shared library.
		n := d.ldr.NDef()
		for i := 1; i < n; i++ {
			s := loader.Sym(i)
			if !d.ldr.IsDup(s) {
				d.mark(s, 0)
			}
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
				d.ReadRelocs(exportsIdx)
				for i := 0; i < len(d.rtmp); i++ {
					d.mark(d.rtmp[i].Sym, 0)
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

	// DWARF constant DIE symbols are not referenced, but needed by
	// the dwarf pass.
	if !*FlagW {
		for _, lib := range d.ctxt.Library {
			names = append(names, dwarf.ConstInfoPrefix+lib.Pkg)
		}
	}

	for _, name := range names {
		// Mark symbol as a data/ABI0 symbol.
		d.mark(d.ldr.Lookup(name, 0), 0)
		// Also mark any Go functions (internal ABI).
		d.mark(d.ldr.Lookup(name, sym.SymVerABIInternal), 0)
	}
}

func (d *deadcodePass2) flood() {
	symRelocs := []loader.Reloc{}
	auxSyms := []loader.Sym{}
	for !d.wq.empty() {
		symIdx := d.wq.pop()

		d.reflectSeen = d.reflectSeen || d.ldr.IsReflectMethod(symIdx)

		relocs := d.ldr.Relocs(symIdx)
		symRelocs = relocs.ReadAll(symRelocs)

		if d.ldr.IsGoType(symIdx) {
			p := d.ldr.Data(symIdx)
			if len(p) != 0 && decodetypeKind(d.ctxt.Arch, p)&kindMask == kindInterface {
				for _, sig := range d.decodeIfaceMethods2(d.ldr, d.ctxt.Arch, symIdx, symRelocs) {
					if d.ctxt.Debugvlog > 1 {
						d.ctxt.Logf("reached iface method: %s\n", sig)
					}
					d.ifaceMethod[sig] = true
				}
			}
		}

		var methods []methodref2
		for i := 0; i < relocs.Count; i++ {
			r := symRelocs[i]
			if r.Type == objabi.R_WEAKADDROFF {
				continue
			}
			if r.Type == objabi.R_METHODOFF {
				if i+2 >= relocs.Count {
					panic("expect three consecutive R_METHODOFF relocs")
				}
				methods = append(methods, methodref2{src: symIdx, r: i})
				i += 2
				continue
			}
			if r.Type == objabi.R_USETYPE {
				// type symbol used for DWARF. we need to load the symbol but it may not
				// be otherwise reachable in the program.
				// do nothing for now as we still load all type symbols.
				continue
			}
			d.mark(r.Sym, symIdx)
		}
		auxSyms = d.ldr.ReadAuxSyms(symIdx, auxSyms)
		for i := 0; i < len(auxSyms); i++ {
			d.mark(auxSyms[i], symIdx)
		}
		// Some host object symbols have an outer object, which acts like a
		// "carrier" symbol, or it holds all the symbols for a particular
		// section. We need to mark all "referenced" symbols from that carrier,
		// so we make sure we're pulling in all outer symbols, and their sub
		// symbols. This is not ideal, and these carrier/section symbols could
		// be removed.
		d.mark(d.ldr.OuterSym(symIdx), symIdx)
		d.mark(d.ldr.SubSym(symIdx), symIdx)

		if len(methods) != 0 {
			// Decode runtime type information for type methods
			// to help work out which methods can be called
			// dynamically via interfaces.
			methodsigs := d.decodetypeMethods2(d.ldr, d.ctxt.Arch, symIdx, symRelocs)
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

func (d *deadcodePass2) mark(symIdx, parent loader.Sym) {
	if symIdx != 0 && !d.ldr.Reachable.Has(symIdx) {
		d.wq.push(symIdx)
		d.ldr.Reachable.Set(symIdx)
		if d.ctxt.Reachparent != nil {
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

func (d *deadcodePass2) markMethod(m methodref2) {
	d.ReadRelocs(m.src)
	d.mark(d.rtmp[m.r].Sym, m.src)
	d.mark(d.rtmp[m.r+1].Sym, m.src)
	d.mark(d.rtmp[m.r+2].Sym, m.src)
}

func deadcode2(ctxt *Link) {
	ldr := ctxt.loader
	d := deadcodePass2{ctxt: ctxt, ldr: ldr}
	d.init()
	d.flood()

	callSym := ldr.Lookup("reflect.Value.Call", sym.SymVerABIInternal)
	methSym := ldr.Lookup("reflect.Value.Method", sym.SymVerABIInternal)
	if ctxt.DynlinkingGo() {
		// Exported methods may satisfy interfaces we don't know
		// about yet when dynamically linking.
		d.reflectSeen = true
	}

	for {
		// Methods might be called via reflection. Give up on
		// static analysis, mark all exported methods of
		// all reachable types as reachable.
		d.reflectSeen = d.reflectSeen || (callSym != 0 && ldr.Reachable.Has(callSym)) || (methSym != 0 && ldr.Reachable.Has(methSym))

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
				if relocs.Count > 0 && ldr.Reachable.Has(relocs.At(0).Sym) {
					ldr.Reachable.Set(s)
				}
			}
		}
	}
}

// methodref2 holds the relocations from a receiver type symbol to its
// method. There are three relocations, one for each of the fields in
// the reflect.method struct: mtyp, ifn, and tfn.
type methodref2 struct {
	m   methodsig
	src loader.Sym // receiver type symbol
	r   int        // the index of R_METHODOFF relocations
}

func (m methodref2) isExported() bool {
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
func (d *deadcodePass2) decodeMethodSig2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc, off, size, count int) []methodsig {
	var buf bytes.Buffer
	var methods []methodsig
	for i := 0; i < count; i++ {
		buf.WriteString(decodetypeName2(ldr, symIdx, symRelocs, off))
		mtypSym := decodeRelocSym2(ldr, symIdx, symRelocs, int32(off+4))
		// FIXME: add some sort of caching here, since we may see some of the
		// same symbols over time for param types.
		d.ReadRelocs(mtypSym)
		mp := ldr.Data(mtypSym)

		buf.WriteRune('(')
		inCount := decodetypeFuncInCount(arch, mp)
		for i := 0; i < inCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			a := d.decodetypeFuncInType2(ldr, arch, mtypSym, d.rtmp, i)
			buf.WriteString(ldr.SymName(a))
		}
		buf.WriteString(") (")
		outCount := decodetypeFuncOutCount(arch, mp)
		for i := 0; i < outCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			a := d.decodetypeFuncOutType2(ldr, arch, mtypSym, d.rtmp, i)
			buf.WriteString(ldr.SymName(a))
		}
		buf.WriteRune(')')

		off += size
		methods = append(methods, methodsig(buf.String()))
		buf.Reset()
	}
	return methods
}

func (d *deadcodePass2) decodeIfaceMethods2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc) []methodsig {
	p := ldr.Data(symIdx)
	if decodetypeKind(arch, p)&kindMask != kindInterface {
		panic(fmt.Sprintf("symbol %q is not an interface", ldr.SymName(symIdx)))
	}
	rel := decodeReloc2(ldr, symIdx, symRelocs, int32(commonsize(arch)+arch.PtrSize))
	if rel.Sym == 0 {
		return nil
	}
	if rel.Sym != symIdx {
		panic(fmt.Sprintf("imethod slice pointer in %q leads to a different symbol", ldr.SymName(symIdx)))
	}
	off := int(rel.Add) // array of reflect.imethod values
	numMethods := int(decodetypeIfaceMethodCount(arch, p))
	sizeofIMethod := 4 + 4
	return d.decodeMethodSig2(ldr, arch, symIdx, symRelocs, off, sizeofIMethod, numMethods)
}

func (d *deadcodePass2) decodetypeMethods2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc) []methodsig {
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
	return d.decodeMethodSig2(ldr, arch, symIdx, symRelocs, off, sizeofMethod, mcount)
}

func decodeReloc2(ldr *loader.Loader, symIdx loader.Sym, symRelocs []loader.Reloc, off int32) loader.Reloc {
	for j := 0; j < len(symRelocs); j++ {
		rel := symRelocs[j]
		if rel.Off == off {
			return rel
		}
	}
	return loader.Reloc{}
}

func decodeRelocSym2(ldr *loader.Loader, symIdx loader.Sym, symRelocs []loader.Reloc, off int32) loader.Sym {
	return decodeReloc2(ldr, symIdx, symRelocs, off).Sym
}

// decodetypeName2 decodes the name from a reflect.name.
func decodetypeName2(ldr *loader.Loader, symIdx loader.Sym, symRelocs []loader.Reloc, off int) string {
	r := decodeRelocSym2(ldr, symIdx, symRelocs, int32(off))
	if r == 0 {
		return ""
	}

	data := ldr.Data(r)
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func (d *deadcodePass2) decodetypeFuncInType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc, i int) loader.Sym {
	uadd := commonsize(arch) + 4
	if arch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(arch, ldr.Data(symIdx)) {
		uadd += uncommonSize()
	}
	return decodeRelocSym2(ldr, symIdx, symRelocs, int32(uadd+i*arch.PtrSize))
}

func (d *deadcodePass2) decodetypeFuncOutType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc, i int) loader.Sym {
	return d.decodetypeFuncInType2(ldr, arch, symIdx, symRelocs, i+decodetypeFuncInCount(arch, ldr.Data(symIdx)))
}

// readRelocs reads the relocations for the specified symbol into the
// deadcode relocs work array. Use with care, since the work array
// is a singleton.
func (d *deadcodePass2) ReadRelocs(symIdx loader.Sym) {
	relocs := d.ldr.Relocs(symIdx)
	d.rtmp = relocs.ReadAll(d.rtmp)
}
