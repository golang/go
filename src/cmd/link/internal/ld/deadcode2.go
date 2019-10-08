// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/objfile"
	"cmd/link/internal/sym"
	"fmt"
	"strings"
	"unicode"
)

var _ = fmt.Print

// TODO:
// - Shared object support:
//   It basically marks everything. We could consider using
//   a different mechanism to represent it.
// - Field tracking support:
//   It needs to record from where the symbol is referenced.
// - Debug output:
//   Emit messages about which symbols are kept or deleted.

type workQueue []objfile.Sym

func (q *workQueue) push(i objfile.Sym) { *q = append(*q, i) }
func (q *workQueue) pop() objfile.Sym   { i := (*q)[len(*q)-1]; *q = (*q)[:len(*q)-1]; return i }
func (q *workQueue) empty() bool        { return len(*q) == 0 }

type deadcodePass2 struct {
	ctxt   *Link
	loader *objfile.Loader
	wq     workQueue

	ifaceMethod     map[methodsig]bool // methods declared in reached interfaces
	markableMethods []methodref2       // methods of reached types
	reflectSeen     bool               // whether we have seen a reflect method call
}

func (d *deadcodePass2) init() {
	d.loader.InitReachable()
	d.ifaceMethod = make(map[methodsig]bool)

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
			exportsIdx := d.loader.Lookup("go.plugin.exports", 0)
			if exportsIdx != 0 {
				relocs := d.loader.Relocs(exportsIdx)
				for i := 0; i < relocs.Count; i++ {
					d.mark(relocs.At(i).Sym)
				}
			}
		}
	}
	for _, s := range dynexp {
		d.mark(d.loader.Lookup(s.Name, int(s.Version)))
	}

	for _, name := range names {
		// Mark symbol as an data/ABI0 symbol.
		d.mark(d.loader.Lookup(name, 0))
		// Also mark any Go functions (internal ABI).
		d.mark(d.loader.Lookup(name, sym.SymVerABIInternal))
	}
}

func (d *deadcodePass2) flood() {
	for !d.wq.empty() {
		symIdx := d.wq.pop()

		d.reflectSeen = d.reflectSeen || d.loader.IsReflectMethod(symIdx)

		name := d.loader.RawSymName(symIdx)
		if strings.HasPrefix(name, "type.") && name[5] != '.' { // TODO: use an attribute instead of checking name
			p := d.loader.Data(symIdx)
			if len(p) != 0 && decodetypeKind(d.ctxt.Arch, p)&kindMask == kindInterface {
				for _, sig := range decodeIfaceMethods2(d.loader, d.ctxt.Arch, symIdx) {
					if d.ctxt.Debugvlog > 1 {
						d.ctxt.Logf("reached iface method: %s\n", sig)
					}
					d.ifaceMethod[sig] = true
				}
			}
		}

		var methods []methodref2
		relocs := d.loader.Relocs(symIdx)
		for i := 0; i < relocs.Count; i++ {
			r := relocs.At(i)
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
			d.mark(r.Sym)
		}
		naux := d.loader.NAux(symIdx)
		for i := 0; i < naux; i++ {
			d.mark(d.loader.AuxSym(symIdx, i))
		}

		if len(methods) != 0 {
			// Decode runtime type information for type methods
			// to help work out which methods can be called
			// dynamically via interfaces.
			methodsigs := decodetypeMethods2(d.loader, d.ctxt.Arch, symIdx)
			if len(methods) != len(methodsigs) {
				panic(fmt.Sprintf("%q has %d method relocations for %d methods", d.loader.SymName(symIdx), len(methods), len(methodsigs)))
			}
			for i, m := range methodsigs {
				methods[i].m = m
			}
			d.markableMethods = append(d.markableMethods, methods...)
		}
	}
}

func (d *deadcodePass2) mark(symIdx objfile.Sym) {
	if symIdx != 0 && !d.loader.Reachable.Has(symIdx) {
		d.wq.push(symIdx)
		d.loader.Reachable.Set(symIdx)
	}
}

func (d *deadcodePass2) markMethod(m methodref2) {
	relocs := d.loader.Relocs(m.src)
	d.mark(relocs.At(m.r).Sym)
	d.mark(relocs.At(m.r + 1).Sym)
	d.mark(relocs.At(m.r + 2).Sym)
}

func deadcode2(ctxt *Link) {
	loader := ctxt.loader
	d := deadcodePass2{ctxt: ctxt, loader: loader}
	d.init()
	d.flood()

	callSym := loader.Lookup("reflect.Value.Call", sym.SymVerABIInternal)
	methSym := loader.Lookup("reflect.Value.Method", sym.SymVerABIInternal)

	if ctxt.DynlinkingGo() {
		// Exported methods may satisfy interfaces we don't know
		// about yet when dynamically linking.
		d.reflectSeen = true
	}

	for {
		// Methods might be called via reflection. Give up on
		// static analysis, mark all exported methods of
		// all reachable types as reachable.
		d.reflectSeen = d.reflectSeen || (callSym != 0 && loader.Reachable.Has(callSym)) || (methSym != 0 && loader.Reachable.Has(methSym))

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

	n := loader.NSym()
	if ctxt.BuildMode != BuildModeShared {
		// Keep a itablink if the symbol it points at is being kept.
		// (When BuildModeShared, always keep itablinks.)
		for i := 1; i < n; i++ {
			s := objfile.Sym(i)
			if strings.HasPrefix(loader.RawSymName(s), "go.itablink.") { // TODO: use an attribute instread of checking name
				relocs := loader.Relocs(s)
				if relocs.Count > 0 && loader.Reachable.Has(relocs.At(0).Sym) {
					loader.Reachable.Set(s)
				}
			}
		}
	}

	// Set reachable attr for now.
	for i := 1; i < n; i++ {
		if loader.Reachable.Has(objfile.Sym(i)) {
			s := loader.Syms[i]
			if s != nil && s.Name != "" {
				s.Attr.Set(sym.AttrReachable, true)
			}
		}
	}
}

// methodref2 holds the relocations from a receiver type symbol to its
// method. There are three relocations, one for each of the fields in
// the reflect.method struct: mtyp, ifn, and tfn.
type methodref2 struct {
	m   methodsig
	src objfile.Sym // receiver type symbol
	r   int         // the index of R_METHODOFF relocations
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
func decodeMethodSig2(loader *objfile.Loader, arch *sys.Arch, symIdx objfile.Sym, off, size, count int) []methodsig {
	var buf bytes.Buffer
	var methods []methodsig
	for i := 0; i < count; i++ {
		buf.WriteString(decodetypeName2(loader, symIdx, off))
		mtypSym := decodeRelocSym2(loader, symIdx, int32(off+4))
		mp := loader.Data(mtypSym)

		buf.WriteRune('(')
		inCount := decodetypeFuncInCount(arch, mp)
		for i := 0; i < inCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			a := decodetypeFuncInType2(loader, arch, mtypSym, i)
			buf.WriteString(loader.SymName(a))
		}
		buf.WriteString(") (")
		outCount := decodetypeFuncOutCount(arch, mp)
		for i := 0; i < outCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			a := decodetypeFuncOutType2(loader, arch, mtypSym, i)
			buf.WriteString(loader.SymName(a))
		}
		buf.WriteRune(')')

		off += size
		methods = append(methods, methodsig(buf.String()))
		buf.Reset()
	}
	return methods
}

func decodeIfaceMethods2(loader *objfile.Loader, arch *sys.Arch, symIdx objfile.Sym) []methodsig {
	p := loader.Data(symIdx)
	if decodetypeKind(arch, p)&kindMask != kindInterface {
		panic(fmt.Sprintf("symbol %q is not an interface", loader.SymName(symIdx)))
	}
	rel := decodeReloc2(loader, symIdx, int32(commonsize(arch)+arch.PtrSize))
	if rel.Sym == 0 {
		return nil
	}
	if rel.Sym != symIdx {
		panic(fmt.Sprintf("imethod slice pointer in %q leads to a different symbol", loader.SymName(symIdx)))
	}
	off := int(rel.Add) // array of reflect.imethod values
	numMethods := int(decodetypeIfaceMethodCount(arch, p))
	sizeofIMethod := 4 + 4
	return decodeMethodSig2(loader, arch, symIdx, off, sizeofIMethod, numMethods)
}

func decodetypeMethods2(loader *objfile.Loader, arch *sys.Arch, symIdx objfile.Sym) []methodsig {
	p := loader.Data(symIdx)
	if !decodetypeHasUncommon(arch, p) {
		panic(fmt.Sprintf("no methods on %q", loader.SymName(symIdx)))
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
	return decodeMethodSig2(loader, arch, symIdx, off, sizeofMethod, mcount)
}

func decodeReloc2(loader *objfile.Loader, symIdx objfile.Sym, off int32) objfile.Reloc {
	relocs := loader.Relocs(symIdx)
	for j := 0; j < relocs.Count; j++ {
		rel := relocs.At(j)
		if rel.Off == off {
			return rel
		}
	}
	return objfile.Reloc{}
}

func decodeRelocSym2(loader *objfile.Loader, symIdx objfile.Sym, off int32) objfile.Sym {
	return decodeReloc2(loader, symIdx, off).Sym
}

// decodetypeName2 decodes the name from a reflect.name.
func decodetypeName2(loader *objfile.Loader, symIdx objfile.Sym, off int) string {
	r := decodeRelocSym2(loader, symIdx, int32(off))
	if r == 0 {
		return ""
	}

	data := loader.Data(r)
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func decodetypeFuncInType2(loader *objfile.Loader, arch *sys.Arch, symIdx objfile.Sym, i int) objfile.Sym {
	uadd := commonsize(arch) + 4
	if arch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(arch, loader.Data(symIdx)) {
		uadd += uncommonSize()
	}
	return decodeRelocSym2(loader, symIdx, int32(uadd+i*arch.PtrSize))
}

func decodetypeFuncOutType2(loader *objfile.Loader, arch *sys.Arch, symIdx objfile.Sym, i int) objfile.Sym {
	return decodetypeFuncInType2(loader, arch, symIdx, i+decodetypeFuncInCount(arch, loader.Data(symIdx)))
}
