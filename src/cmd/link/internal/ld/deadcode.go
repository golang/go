// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
	"cmd/internal/sys"
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
		ctxt.Logf("%5.2f deadcode\n", obj.Cputime())
	}

	d := &deadcodepass{
		ctxt:        ctxt,
		ifaceMethod: make(map[methodsig]bool),
	}

	// First, flood fill any symbols directly reachable in the call
	// graph from *flagEntrySymbol. Ignore all methods not directly called.
	d.init()
	d.flood()

	callSym := ctxt.Syms.ROLookup("reflect.Value.Call", 0)
	methSym := ctxt.Syms.ROLookup("reflect.Value.Method", 0)
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

	if Buildmode != BuildmodeShared {
		// Keep a itablink if the symbol it points at is being kept.
		// (When BuildmodeShared, always keep itablinks.)
		for _, s := range ctxt.Syms.Allsym {
			if strings.HasPrefix(s.Name, "go.itablink.") {
				s.Attr.Set(AttrReachable, len(s.R) == 1 && s.R[0].Sym.Attr.Reachable())
			}
		}
	}

	// Remove dead text but keep file information (z symbols).
	textp := make([]*Symbol, 0, len(ctxt.Textp))
	for _, s := range ctxt.Textp {
		if s.Attr.Reachable() {
			textp = append(textp, s)
		}
	}
	ctxt.Textp = textp
}

var markextra = []string{
	"runtime.morestack",
	"runtime.morestackx",
	"runtime.morestack00",
	"runtime.morestack10",
	"runtime.morestack01",
	"runtime.morestack11",
	"runtime.morestack8",
	"runtime.morestack16",
	"runtime.morestack24",
	"runtime.morestack32",
	"runtime.morestack40",
	"runtime.morestack48",

	// on arm, lock in the div/mod helpers too
	"_div",
	"_divu",
	"_mod",
	"_modu",
}

// methodref holds the relocations from a receiver type symbol to its
// method. There are three relocations, one for each of the fields in
// the reflect.method struct: mtyp, ifn, and tfn.
type methodref struct {
	m   methodsig
	src *Symbol   // receiver type symbol
	r   [3]*Reloc // R_METHODOFF relocations to fields of runtime.method
}

func (m methodref) ifn() *Symbol { return m.r[1].Sym }

func (m methodref) isExported() bool {
	for _, r := range m.m {
		return unicode.IsUpper(r)
	}
	panic("methodref has no signature")
}

// deadcodepass holds state for the deadcode flood fill.
type deadcodepass struct {
	ctxt            *Link
	markQueue       []*Symbol          // symbols to flood fill in next pass
	ifaceMethod     map[methodsig]bool // methods declared in reached interfaces
	markableMethods []methodref        // methods of reached types
	reflectMethod   bool
}

func (d *deadcodepass) cleanupReloc(r *Reloc) {
	if r.Sym.Attr.Reachable() {
		r.Type = obj.R_ADDROFF
	} else {
		if d.ctxt.Debugvlog > 1 {
			d.ctxt.Logf("removing method %s\n", r.Sym.Name)
		}
		r.Sym = nil
		r.Siz = 0
	}
}

// mark appends a symbol to the mark queue for flood filling.
func (d *deadcodepass) mark(s, parent *Symbol) {
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
	s.Attr |= AttrReachable
	s.Reachparent = parent
	d.markQueue = append(d.markQueue, s)
}

// markMethod marks a method as reachable.
func (d *deadcodepass) markMethod(m methodref) {
	for _, r := range m.r {
		d.mark(r.Sym, m.src)
		r.Type = obj.R_ADDROFF
	}
}

// init marks all initial symbols as reachable.
// In a typical binary, this is *flagEntrySymbol.
func (d *deadcodepass) init() {
	var names []string

	if SysArch.Family == sys.ARM {
		// mark some functions that are only referenced after linker code editing
		if obj.GOARM == 5 {
			names = append(names, "_sfloat")
		}
		names = append(names, "runtime.read_tls_fallback")
	}

	if Buildmode == BuildmodeShared {
		// Mark all symbols defined in this library as reachable when
		// building a shared library.
		for _, s := range d.ctxt.Syms.Allsym {
			if s.Type != 0 && s.Type != obj.SDYNIMPORT {
				d.mark(s, nil)
			}
		}
	} else {
		// In a normal binary, start at main.main and the init
		// functions and mark what is reachable from there.
		names = append(names, *flagEntrySymbol)
		if *FlagLinkshared && (Buildmode == BuildmodeExe || Buildmode == BuildmodePIE) {
			names = append(names, "main.main", "main.init")
		} else if Buildmode == BuildmodePlugin {
			names = append(names, *flagPluginPath+".init", *flagPluginPath+".main", "go.plugin.tabs")

			// We don't keep the go.plugin.exports symbol,
			// but we do keep the symbols it refers to.
			exports := d.ctxt.Syms.ROLookup("go.plugin.exports", 0)
			if exports != nil {
				for _, r := range exports.R {
					d.mark(r.Sym, nil)
				}
			}
		}
		for _, name := range markextra {
			names = append(names, name)
		}
		for _, s := range dynexp {
			d.mark(s, nil)
		}
	}

	for _, name := range names {
		d.mark(d.ctxt.Syms.ROLookup(name, 0), nil)
	}
}

// flood flood fills symbols reachable from the markQueue symbols.
// As it goes, it collects methodref and interface method declarations.
func (d *deadcodepass) flood() {
	for len(d.markQueue) > 0 {
		s := d.markQueue[0]
		d.markQueue = d.markQueue[1:]
		if s.Type == obj.STEXT {
			if d.ctxt.Debugvlog > 1 {
				d.ctxt.Logf("marktext %s\n", s.Name)
			}
			if s.FuncInfo != nil {
				for _, a := range s.FuncInfo.Autom {
					d.mark(a.Gotype, s)
				}
			}

		}

		if strings.HasPrefix(s.Name, "type.") && s.Name[5] != '.' {
			if len(s.P) == 0 {
				// Probably a bug. The undefined symbol check
				// later will give a better error than deadcode.
				continue
			}
			if decodetypeKind(s)&kindMask == kindInterface {
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
		for i := 0; i < len(s.R); i++ {
			r := &s.R[i]
			if r.Sym == nil {
				continue
			}
			if r.Type == obj.R_WEAKADDROFF {
				// An R_WEAKADDROFF relocation is not reason
				// enough to mark the pointed-to symbol as
				// reachable.
				continue
			}
			if r.Type != obj.R_METHODOFF {
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
