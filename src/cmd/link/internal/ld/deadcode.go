// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"fmt"
	"strings"
	"unicode"
)

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
//	3. reflect.Value.Call, .Method, or reflect.Method.Func
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
//	- reflect.Value.Call is reachable
//	- reflect.Value.Method is reachable
// 	- reflect.Type.Method or MethodByName is called.
// If any of these happen, all bets are off and all exported methods
// of reachable types are marked reachable.
//
// Any unreached text symbols are removed from ctxt.Textp.
func deadcode(ctxt *Link) {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("deadcode\n")
	}

	if *flagNewobj {
		deadcode2(ctxt)
		return
	}

	d := &deadcodepass{
		ctxt:        ctxt,
		ifaceMethod: make(map[methodsig]bool),
	}

	// First, flood fill any symbols directly reachable in the call
	// graph from *flagEntrySymbol. Ignore all methods not directly called.
	d.init()
	d.flood()

	callSym := ctxt.Syms.ROLookup("reflect.Value.Call", sym.SymVerABIInternal)
	methSym := ctxt.Syms.ROLookup("reflect.Value.Method", sym.SymVerABIInternal)
	reflectSeen := false

	if ctxt.DynlinkingGo() {
		// Exported methods may satisfy interfaces we don't know
		// about yet when dynamically linking.
		reflectSeen = true
	}

	for {
		if !reflectSeen {
			if d.reflectMethod || (callSym != nil && callSym.Attr.Reachable()) || (methSym != nil && methSym.Attr.Reachable()) {
				// Methods might be called via reflection. Give up on
				// static analysis, mark all exported methods of
				// all reachable types as reachable.
				reflectSeen = true
			}
		}

		// Mark all methods that could satisfy a discovered
		// interface as reachable. We recheck old marked interfaces
		// as new types (with new methods) may have been discovered
		// in the last pass.
		var rem []methodref
		for _, m := range d.markableMethods {
			if (reflectSeen && m.isExported()) || d.ifaceMethod[m.m] {
				d.markMethod(m)
			} else {
				rem = append(rem, m)
			}
		}
		d.markableMethods = rem

		if len(d.markQueue) == 0 {
			// No new work was discovered. Done.
			break
		}
		d.flood()
	}

	// Remove all remaining unreached R_METHODOFF relocations.
	for _, m := range d.markableMethods {
		for _, r := range m.r {
			d.cleanupReloc(r)
		}
	}

	if ctxt.BuildMode != BuildModeShared {
		// Keep a itablink if the symbol it points at is being kept.
		// (When BuildModeShared, always keep itablinks.)
		for _, s := range ctxt.Syms.Allsym {
			if strings.HasPrefix(s.Name, "go.itablink.") {
				s.Attr.Set(sym.AttrReachable, len(s.R) == 1 && s.R[0].Sym.Attr.Reachable())
			}
		}
	}

	addToTextp(ctxt)
}

func addToTextp(ctxt *Link) {
	// Remove dead text but keep file information (z symbols).
	textp := []*sym.Symbol{}
	for _, s := range ctxt.Textp {
		if s.Attr.Reachable() {
			textp = append(textp, s)
		}
	}

	// Put reachable text symbols into Textp.
	// do it in postorder so that packages are laid down in dependency order
	// internal first, then everything else
	ctxt.Library = postorder(ctxt.Library)
	for _, doInternal := range [2]bool{true, false} {
		for _, lib := range ctxt.Library {
			if isRuntimeDepPkg(lib.Pkg) != doInternal {
				continue
			}
			libtextp := lib.Textp[:0]
			for _, s := range lib.Textp {
				if s.Attr.Reachable() {
					textp = append(textp, s)
					libtextp = append(libtextp, s)
					if s.Unit != nil {
						s.Unit.Textp = append(s.Unit.Textp, s)
					}
				}
			}
			for _, s := range lib.DupTextSyms {
				if s.Attr.Reachable() && !s.Attr.OnList() {
					textp = append(textp, s)
					libtextp = append(libtextp, s)
					if s.Unit != nil {
						s.Unit.Textp = append(s.Unit.Textp, s)
					}
					s.Attr |= sym.AttrOnList
					// dupok symbols may be defined in multiple packages. its
					// associated package is chosen sort of arbitrarily (the
					// first containing package that the linker loads). canonicalize
					// it here to the package with which it will be laid down
					// in text.
					s.File = objabi.PathToPrefix(lib.Pkg)
				}
			}
			lib.Textp = libtextp
		}
	}
	ctxt.Textp = textp

	if len(ctxt.Shlibs) > 0 {
		// We might have overwritten some functions above (this tends to happen for the
		// autogenerated type equality/hashing functions) and we don't want to generated
		// pcln table entries for these any more so remove them from Textp.
		textp := make([]*sym.Symbol, 0, len(ctxt.Textp))
		for _, s := range ctxt.Textp {
			if s.Type != sym.SDYNIMPORT {
				textp = append(textp, s)
			}
		}
		ctxt.Textp = textp
	}
}

// methodref holds the relocations from a receiver type symbol to its
// method. There are three relocations, one for each of the fields in
// the reflect.method struct: mtyp, ifn, and tfn.
type methodref struct {
	m   methodsig
	src *sym.Symbol   // receiver type symbol
	r   [3]*sym.Reloc // R_METHODOFF relocations to fields of runtime.method
}

func (m methodref) ifn() *sym.Symbol { return m.r[1].Sym }

func (m methodref) isExported() bool {
	for _, r := range m.m {
		return unicode.IsUpper(r)
	}
	panic("methodref has no signature")
}

// deadcodepass holds state for the deadcode flood fill.
type deadcodepass struct {
	ctxt            *Link
	markQueue       []*sym.Symbol      // symbols to flood fill in next pass
	ifaceMethod     map[methodsig]bool // methods declared in reached interfaces
	markableMethods []methodref        // methods of reached types
	reflectMethod   bool
}

func (d *deadcodepass) cleanupReloc(r *sym.Reloc) {
	if r.Sym.Attr.Reachable() {
		r.Type = objabi.R_ADDROFF
	} else {
		if d.ctxt.Debugvlog > 1 {
			d.ctxt.Logf("removing method %s\n", r.Sym.Name)
		}
		r.Sym = nil
		r.Siz = 0
	}
}

// mark appends a symbol to the mark queue for flood filling.
func (d *deadcodepass) mark(s, parent *sym.Symbol) {
	if s == nil || s.Attr.Reachable() {
		return
	}
	if s.Attr.ReflectMethod() {
		d.reflectMethod = true
	}
	if *flagDumpDep {
		p := "_"
		if parent != nil {
			p = parent.Name
		}
		fmt.Printf("%s -> %s\n", p, s.Name)
	}
	s.Attr |= sym.AttrReachable
	if d.ctxt.Reachparent != nil {
		d.ctxt.Reachparent[s] = parent
	}
	d.markQueue = append(d.markQueue, s)
}

// markMethod marks a method as reachable.
func (d *deadcodepass) markMethod(m methodref) {
	for _, r := range m.r {
		d.mark(r.Sym, m.src)
		r.Type = objabi.R_ADDROFF
	}
}

// init marks all initial symbols as reachable.
// In a typical binary, this is *flagEntrySymbol.
func (d *deadcodepass) init() {
	var names []string

	if d.ctxt.BuildMode == BuildModeShared {
		// Mark all symbols defined in this library as reachable when
		// building a shared library.
		for _, s := range d.ctxt.Syms.Allsym {
			if s.Type != 0 && s.Type != sym.SDYNIMPORT {
				d.mark(s, nil)
			}
		}
	} else {
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
				exports := d.ctxt.Syms.ROLookup("go.plugin.exports", 0)
				if exports != nil {
					for i := range exports.R {
						d.mark(exports.R[i].Sym, nil)
					}
				}
			}
		}
		for _, s := range dynexp {
			d.mark(s, nil)
		}
	}

	for _, name := range names {
		// Mark symbol as a data/ABI0 symbol.
		d.mark(d.ctxt.Syms.ROLookup(name, 0), nil)
		// Also mark any Go functions (internal ABI).
		d.mark(d.ctxt.Syms.ROLookup(name, sym.SymVerABIInternal), nil)
	}
}

// flood fills symbols reachable from the markQueue symbols.
// As it goes, it collects methodref and interface method declarations.
func (d *deadcodepass) flood() {
	for len(d.markQueue) > 0 {
		s := d.markQueue[0]
		d.markQueue = d.markQueue[1:]
		if s.Type == sym.STEXT {
			if d.ctxt.Debugvlog > 1 {
				d.ctxt.Logf("marktext %s\n", s.Name)
			}
		}

		if strings.HasPrefix(s.Name, "type.") && s.Name[5] != '.' {
			if len(s.P) == 0 {
				// Probably a bug. The undefined symbol check
				// later will give a better error than deadcode.
				continue
			}
			if decodetypeKind(d.ctxt.Arch, s.P)&kindMask == kindInterface {
				for _, sig := range decodeIfaceMethods(d.ctxt.Arch, s) {
					if d.ctxt.Debugvlog > 1 {
						d.ctxt.Logf("reached iface method: %s\n", sig)
					}
					d.ifaceMethod[sig] = true
				}
			}
		}

		mpos := 0 // 0-3, the R_METHODOFF relocs of runtime.uncommontype
		var methods []methodref
		for i := range s.R {
			r := &s.R[i]
			if r.Sym == nil {
				continue
			}
			if r.Type == objabi.R_WEAKADDROFF {
				// An R_WEAKADDROFF relocation is not reason
				// enough to mark the pointed-to symbol as
				// reachable.
				continue
			}
			if r.Sym.Type == sym.SABIALIAS {
				// Patch this relocation through the
				// ABI alias before marking.
				r.Sym = resolveABIAlias(r.Sym)
			}
			if r.Type != objabi.R_METHODOFF {
				d.mark(r.Sym, s)
				continue
			}
			// Collect rtype pointers to methods for
			// later processing in deadcode.
			if mpos == 0 {
				m := methodref{src: s}
				m.r[0] = r
				methods = append(methods, m)
			} else {
				methods[len(methods)-1].r[mpos] = r
			}
			mpos++
			if mpos == len(methodref{}.r) {
				mpos = 0
			}
		}
		if len(methods) > 0 {
			// Decode runtime type information for type methods
			// to help work out which methods can be called
			// dynamically via interfaces.
			methodsigs := decodetypeMethods(d.ctxt.Arch, s)
			if len(methods) != len(methodsigs) {
				panic(fmt.Sprintf("%q has %d method relocations for %d methods", s.Name, len(methods), len(methodsigs)))
			}
			for i, m := range methodsigs {
				name := string(m)
				name = name[:strings.Index(name, "(")]
				if !strings.HasSuffix(methods[i].ifn().Name, name) {
					panic(fmt.Sprintf("%q relocation for %q does not match method %q", s.Name, methods[i].ifn().Name, name))
				}
				methods[i].m = m
			}
			d.markableMethods = append(d.markableMethods, methods...)
		}

		if s.FuncInfo != nil {
			for i := range s.FuncInfo.Funcdata {
				d.mark(s.FuncInfo.Funcdata[i], s)
			}
		}
		d.mark(s.Gotype, s)
		d.mark(s.Sub, s)
		d.mark(s.Outer, s)
	}
}
