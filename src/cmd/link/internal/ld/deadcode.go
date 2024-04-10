// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"internal/abi"
	"internal/buildcfg"
	"strings"
	"unicode"
)

var _ = fmt.Print

type deadcodePass struct {
	ctxt *Link
	ldr  *loader.Loader
	wq   heap // work queue, using min-heap for better locality

	ifaceMethod        map[methodsig]bool // methods called from reached interface call sites
	genericIfaceMethod map[string]bool    // names of methods called from reached generic interface call sites
	markableMethods    []methodref        // methods of reached types
	reflectSeen        bool               // whether we have seen a reflect method call
	dynlink            bool

	methodsigstmp []methodsig // scratch buffer for decoding method signatures
	pkginits      []loader.Sym
	mapinitnoop   loader.Sym
}

func (d *deadcodePass) init() {
	d.ldr.InitReachable()
	d.ifaceMethod = make(map[methodsig]bool)
	d.genericIfaceMethod = make(map[string]bool)
	if buildcfg.Experiment.FieldTrack {
		d.ldr.Reachparent = make([]loader.Sym, d.ldr.NSym())
	}
	d.dynlink = d.ctxt.DynlinkingGo()

	if d.ctxt.BuildMode == BuildModeShared {
		// Mark all symbols defined in this library as reachable when
		// building a shared library.
		n := d.ldr.NDef()
		for i := 1; i < n; i++ {
			s := loader.Sym(i)
			d.mark(s, 0)
		}
		d.mark(d.ctxt.mainInittasks, 0)
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
	}
	// runtime.unreachableMethod is a function that will throw if called.
	// We redirect unreachable methods to it.
	names = append(names, "runtime.unreachableMethod")
	if d.ctxt.BuildMode == BuildModePlugin {
		names = append(names, objabi.PathToPrefix(*flagPluginPath)+"..inittask", objabi.PathToPrefix(*flagPluginPath)+".main", "go:plugin.tabs")

		// We don't keep the go.plugin.exports symbol,
		// but we do keep the symbols it refers to.
		exportsIdx := d.ldr.Lookup("go:plugin.exports", 0)
		if exportsIdx != 0 {
			relocs := d.ldr.Relocs(exportsIdx)
			for i := 0; i < relocs.Count(); i++ {
				d.mark(relocs.At(i).Sym(), 0)
			}
		}
	}

	if d.ctxt.Debugvlog > 1 {
		d.ctxt.Logf("deadcode start names: %v\n", names)
	}

	for _, name := range names {
		// Mark symbol as a data/ABI0 symbol.
		d.mark(d.ldr.Lookup(name, 0), 0)
		if abiInternalVer != 0 {
			// Also mark any Go functions (internal ABI).
			d.mark(d.ldr.Lookup(name, abiInternalVer), 0)
		}
	}

	// All dynamic exports are roots.
	for _, s := range d.ctxt.dynexp {
		if d.ctxt.Debugvlog > 1 {
			d.ctxt.Logf("deadcode start dynexp: %s<%d>\n", d.ldr.SymName(s), d.ldr.SymVersion(s))
		}
		d.mark(s, 0)
	}

	d.mapinitnoop = d.ldr.Lookup("runtime.mapinitnoop", abiInternalVer)
	if d.mapinitnoop == 0 {
		panic("could not look up runtime.mapinitnoop")
	}
	if d.ctxt.mainInittasks != 0 {
		d.mark(d.ctxt.mainInittasks, 0)
	}
}

func (d *deadcodePass) flood() {
	var methods []methodref
	for !d.wq.empty() {
		symIdx := d.wq.pop()

		// Methods may be called via reflection. Give up on static analysis,
		// and mark all exported methods of all reachable types as reachable.
		d.reflectSeen = d.reflectSeen || d.ldr.IsReflectMethod(symIdx)

		isgotype := d.ldr.IsGoType(symIdx)
		relocs := d.ldr.Relocs(symIdx)
		var usedInIface bool

		if isgotype {
			if d.dynlink {
				// When dynamic linking, a type may be passed across DSO
				// boundary and get converted to interface at the other side.
				d.ldr.SetAttrUsedInIface(symIdx, true)
			}
			usedInIface = d.ldr.AttrUsedInIface(symIdx)
		}

		methods = methods[:0]
		for i := 0; i < relocs.Count(); i++ {
			r := relocs.At(i)
			if r.Weak() {
				convertWeakToStrong := false
				// When build with "-linkshared", we can't tell if the
				// interface method in itab will be used or not.
				// Ignore the weak attribute.
				if d.ctxt.linkShared && d.ldr.IsItab(symIdx) {
					convertWeakToStrong = true
				}
				// If the program uses plugins, we can no longer treat
				// relocs from pkg init functions to outlined map init
				// fragments as weak, since doing so can cause package
				// init clashes between the main program and the
				// plugin. See #62430 for more details.
				if d.ctxt.canUsePlugins && r.Type().IsDirectCall() {
					convertWeakToStrong = true
				}
				if !convertWeakToStrong {
					// skip this reloc
					continue
				}
			}
			t := r.Type()
			switch t {
			case objabi.R_METHODOFF:
				if i+2 >= relocs.Count() {
					panic("expect three consecutive R_METHODOFF relocs")
				}
				if usedInIface {
					methods = append(methods, methodref{src: symIdx, r: i})
					// The method descriptor is itself a type descriptor, and
					// it can be used to reach other types, e.g. by using
					// reflect.Type.Method(i).Type.In(j). We need to traverse
					// its child types with UsedInIface set. (See also the
					// comment below.)
					rs := r.Sym()
					if !d.ldr.AttrUsedInIface(rs) {
						d.ldr.SetAttrUsedInIface(rs, true)
						if d.ldr.AttrReachable(rs) {
							d.ldr.SetAttrReachable(rs, false)
							d.mark(rs, symIdx)
						}
					}
				}
				i += 2
				continue
			case objabi.R_USETYPE:
				// type symbol used for DWARF. we need to load the symbol but it may not
				// be otherwise reachable in the program.
				// do nothing for now as we still load all type symbols.
				continue
			case objabi.R_USEIFACE:
				// R_USEIFACE is a marker relocation that tells the linker the type is
				// converted to an interface, i.e. should have UsedInIface set. See the
				// comment below for why we need to unset the Reachable bit and re-mark it.
				rs := r.Sym()
				if d.ldr.IsItab(rs) {
					// This relocation can also point at an itab, in which case it
					// means "the Type field of that itab".
					rs = decodeItabType(d.ldr, d.ctxt.Arch, rs)
				}
				if !d.ldr.IsGoType(rs) && !d.ctxt.linkShared {
					panic(fmt.Sprintf("R_USEIFACE in %s references %s which is not a type or itab", d.ldr.SymName(symIdx), d.ldr.SymName(rs)))
				}
				if !d.ldr.AttrUsedInIface(rs) {
					d.ldr.SetAttrUsedInIface(rs, true)
					if d.ldr.AttrReachable(rs) {
						d.ldr.SetAttrReachable(rs, false)
						d.mark(rs, symIdx)
					}
				}
				continue
			case objabi.R_USEIFACEMETHOD:
				// R_USEIFACEMETHOD is a marker relocation that marks an interface
				// method as used.
				rs := r.Sym()
				if d.ctxt.linkShared && (d.ldr.SymType(rs) == sym.SDYNIMPORT || d.ldr.SymType(rs) == sym.Sxxx) {
					// Don't decode symbol from shared library (we'll mark all exported methods anyway).
					// We check for both SDYNIMPORT and Sxxx because name-mangled symbols haven't
					// been resolved at this point.
					continue
				}
				m := d.decodeIfaceMethod(d.ldr, d.ctxt.Arch, rs, r.Add())
				if d.ctxt.Debugvlog > 1 {
					d.ctxt.Logf("reached iface method: %v\n", m)
				}
				d.ifaceMethod[m] = true
				continue
			case objabi.R_USENAMEDMETHOD:
				name := d.decodeGenericIfaceMethod(d.ldr, r.Sym())
				if d.ctxt.Debugvlog > 1 {
					d.ctxt.Logf("reached generic iface method: %s\n", name)
				}
				d.genericIfaceMethod[name] = true
				continue // don't mark referenced symbol - it is not needed in the final binary.
			case objabi.R_INITORDER:
				// inittasks has already run, so any R_INITORDER links are now
				// superfluous - the only live inittask records are those which are
				// in a scheduled list somewhere (e.g. runtime.moduledata.inittasks).
				continue
			}
			rs := r.Sym()
			if isgotype && usedInIface && d.ldr.IsGoType(rs) && !d.ldr.AttrUsedInIface(rs) {
				// If a type is converted to an interface, it is possible to obtain an
				// interface with a "child" type of it using reflection (e.g. obtain an
				// interface of T from []chan T). We need to traverse its "child" types
				// with UsedInIface attribute set.
				// When visiting the child type (chan T in the example above), it will
				// have UsedInIface set, so it in turn will mark and (re)visit its children
				// (e.g. T above).
				// We unset the reachable bit here, so if the child type is already visited,
				// it will be visited again.
				// Note that a type symbol can be visited at most twice, one without
				// UsedInIface and one with. So termination is still guaranteed.
				d.ldr.SetAttrUsedInIface(rs, true)
				d.ldr.SetAttrReachable(rs, false)
			}
			d.mark(rs, symIdx)
		}
		naux := d.ldr.NAux(symIdx)
		for i := 0; i < naux; i++ {
			a := d.ldr.Aux(symIdx, i)
			if a.Type() == goobj.AuxGotype {
				// A symbol being reachable doesn't imply we need its
				// type descriptor. Don't mark it.
				continue
			}
			d.mark(a.Sym(), symIdx)
		}
		// Record sym if package init func (here naux != 0 is a cheap way
		// to check first if it is a function symbol).
		if naux != 0 && d.ldr.IsPkgInit(symIdx) {

			d.pkginits = append(d.pkginits, symIdx)
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
				if d.ctxt.Debugvlog > 1 {
					d.ctxt.Logf("markable method: %v of sym %v %s\n", m, symIdx, d.ldr.SymName(symIdx))
				}
			}
			d.markableMethods = append(d.markableMethods, methods...)
		}
	}
}

// mapinitcleanup walks all pkg init functions and looks for weak relocations
// to mapinit symbols that are no longer reachable. It rewrites
// the relocs to target a new no-op routine in the runtime.
func (d *deadcodePass) mapinitcleanup() {
	for _, idx := range d.pkginits {
		relocs := d.ldr.Relocs(idx)
		var su *loader.SymbolBuilder
		for i := 0; i < relocs.Count(); i++ {
			r := relocs.At(i)
			rs := r.Sym()
			if r.Weak() && r.Type().IsDirectCall() && !d.ldr.AttrReachable(rs) {
				// double check to make sure target is indeed map.init
				rsn := d.ldr.SymName(rs)
				if !strings.Contains(rsn, "map.init") {
					panic(fmt.Sprintf("internal error: expected map.init sym for weak call reloc, got %s -> %s", d.ldr.SymName(idx), rsn))
				}
				d.ldr.SetAttrReachable(d.mapinitnoop, true)
				if d.ctxt.Debugvlog > 1 {
					d.ctxt.Logf("deadcode: %s rewrite %s ref to %s\n",
						d.ldr.SymName(idx), rsn,
						d.ldr.SymName(d.mapinitnoop))
				}
				if su == nil {
					su = d.ldr.MakeSymbolUpdater(idx)
				}
				su.SetRelocSym(i, d.mapinitnoop)
			}
		}
	}
}

func (d *deadcodePass) mark(symIdx, parent loader.Sym) {
	if symIdx != 0 && !d.ldr.AttrReachable(symIdx) {
		d.wq.push(symIdx)
		d.ldr.SetAttrReachable(symIdx, true)
		if buildcfg.Experiment.FieldTrack && d.ldr.Reachparent[symIdx] == 0 {
			d.ldr.Reachparent[symIdx] = parent
		}
		if *flagDumpDep {
			to := d.ldr.SymName(symIdx)
			if to != "" {
				to = d.dumpDepAddFlags(to, symIdx)
				from := "_"
				if parent != 0 {
					from = d.ldr.SymName(parent)
					from = d.dumpDepAddFlags(from, parent)
				}
				fmt.Printf("%s -> %s\n", from, to)
			}
		}
	}
}

func (d *deadcodePass) dumpDepAddFlags(name string, symIdx loader.Sym) string {
	var flags strings.Builder
	if d.ldr.AttrUsedInIface(symIdx) {
		flags.WriteString("<UsedInIface>")
	}
	if d.ldr.IsReflectMethod(symIdx) {
		flags.WriteString("<ReflectMethod>")
	}
	if flags.Len() > 0 {
		return name + " " + flags.String()
	}
	return name
}

func (d *deadcodePass) markMethod(m methodref) {
	relocs := d.ldr.Relocs(m.src)
	d.mark(relocs.At(m.r).Sym(), m.src)
	d.mark(relocs.At(m.r+1).Sym(), m.src)
	d.mark(relocs.At(m.r+2).Sym(), m.src)
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
//  1. direct call
//  2. through a reachable interface type
//  3. reflect.Value.Method (or MethodByName), or reflect.Type.Method
//     (or MethodByName)
//
// The first case is handled by the flood fill, a directly called method
// is marked as reachable.
//
// The second case is handled by decomposing all reachable interface
// types into method signatures. Each encountered method is compared
// against the interface method signatures, if it matches it is marked
// as reachable. This is extremely conservative, but easy and correct.
//
// The third case is handled by looking for functions that compiler flagged
// as REFLECTMETHOD. REFLECTMETHOD on a function F means that F does a method
// lookup with reflection, but the compiler was not able to statically determine
// the method name.
//
// All functions that call reflect.Value.Method or reflect.Type.Method are REFLECTMETHODs.
// Functions that call reflect.Value.MethodByName or reflect.Type.MethodByName with
// a non-constant argument are REFLECTMETHODs, too. If we find a REFLECTMETHOD,
// we give up on static analysis, and mark all exported methods of all reachable
// types as reachable.
//
// If the argument to MethodByName is a compile-time constant, the compiler
// emits a relocation with the method name. Matching methods are kept in all
// reachable types.
//
// Any unreached text symbols are removed from ctxt.Textp.
func deadcode(ctxt *Link) {
	ldr := ctxt.loader
	d := deadcodePass{ctxt: ctxt, ldr: ldr}
	d.init()
	d.flood()

	if ctxt.DynlinkingGo() {
		// Exported methods may satisfy interfaces we don't know
		// about yet when dynamically linking.
		d.reflectSeen = true
	}

	for {
		// Mark all methods that could satisfy a discovered
		// interface as reachable. We recheck old marked interfaces
		// as new types (with new methods) may have been discovered
		// in the last pass.
		rem := d.markableMethods[:0]
		for _, m := range d.markableMethods {
			if (d.reflectSeen && (m.isExported() || d.dynlink)) || d.ifaceMethod[m.m] || d.genericIfaceMethod[m.m.name] {
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
	if *flagPruneWeakMap {
		d.mapinitcleanup()
	}
}

// methodsig is a typed method signature (name + type).
type methodsig struct {
	name string
	typ  loader.Sym // type descriptor symbol of the function
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
	for _, r := range m.m.name {
		return unicode.IsUpper(r)
	}
	panic("methodref has no signature")
}

// decodeMethodSig decodes an array of method signature information.
// Each element of the array is size bytes. The first 4 bytes is a
// nameOff for the method name, and the next 4 bytes is a typeOff for
// the function type.
//
// Conveniently this is the layout of both runtime.method and runtime.imethod.
func (d *deadcodePass) decodeMethodSig(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, off, size, count int) []methodsig {
	if cap(d.methodsigstmp) < count {
		d.methodsigstmp = append(d.methodsigstmp[:0], make([]methodsig, count)...)
	}
	var methods = d.methodsigstmp[:count]
	for i := 0; i < count; i++ {
		methods[i].name = decodetypeName(ldr, symIdx, relocs, off)
		methods[i].typ = decodeRelocSym(ldr, symIdx, relocs, int32(off+4))
		off += size
	}
	return methods
}

// Decode the method of interface type symbol symIdx at offset off.
func (d *deadcodePass) decodeIfaceMethod(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, off int64) methodsig {
	p := ldr.Data(symIdx)
	if p == nil {
		panic(fmt.Sprintf("missing symbol %q", ldr.SymName(symIdx)))
	}
	if decodetypeKind(arch, p) != abi.Interface {
		panic(fmt.Sprintf("symbol %q is not an interface", ldr.SymName(symIdx)))
	}
	relocs := ldr.Relocs(symIdx)
	var m methodsig
	m.name = decodetypeName(ldr, symIdx, &relocs, int(off))
	m.typ = decodeRelocSym(ldr, symIdx, &relocs, int32(off+4))
	return m
}

// Decode the method name stored in symbol symIdx. The symbol should contain just the bytes of a method name.
func (d *deadcodePass) decodeGenericIfaceMethod(ldr *loader.Loader, symIdx loader.Sym) string {
	return ldr.DataString(symIdx)
}

func (d *deadcodePass) decodetypeMethods(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs) []methodsig {
	p := ldr.Data(symIdx)
	if !decodetypeHasUncommon(arch, p) {
		panic(fmt.Sprintf("no methods on %q", ldr.SymName(symIdx)))
	}
	off := commonsize(arch) // reflect.rtype
	switch decodetypeKind(arch, p) {
	case abi.Struct: // reflect.structType
		off += 4 * arch.PtrSize
	case abi.Pointer: // reflect.ptrType
		off += arch.PtrSize
	case abi.Func: // reflect.funcType
		off += arch.PtrSize // 4 bytes, pointer aligned
	case abi.Slice: // reflect.sliceType
		off += arch.PtrSize
	case abi.Array: // reflect.arrayType
		off += 3 * arch.PtrSize
	case abi.Chan: // reflect.chanType
		off += 2 * arch.PtrSize
	case abi.Map: // reflect.mapType
		off += 4*arch.PtrSize + 8
	case abi.Interface: // reflect.interfaceType
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
