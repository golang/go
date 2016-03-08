// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
	"fmt"
	"strings"
	"unicode"
)

// deadcode marks all reachable symbols.
//
// The basis of the dead code elimination is a flood fill of symbols,
// following their relocations, begining at INITENTRY.
//
// This flood fill is wrapped in logic for pruning unused methods.
// All methods are mentioned by relocations on their receiver's *rtype.
// These relocations are specially defined as R_METHOD by the compiler
// so we can detect and manipulated them here.
//
// There are three ways a method of a reachable type can be invoked:
//
//	1. direct call
//	2. through a reachable interface type
//	3. reflect.Value.Call
//
// The first case is handled by the flood fill, a directly called method
// is marked as reachable.
//
// The second case is handled by decomposing all reachable interface
// types into method signatures. Each encountered method is compared
// against the interface method signatures, if it matches it is marked
// as reachable. This is extremely conservative, but easy and correct.
//
// The third case is handled by looking to see if reflect.Value.Call is
// ever marked reachable. If it is, all bets are off and all exported
// methods of reachable types are marked reachable.
//
// Any unreached text symbols are removed from ctxt.Textp.
func deadcode(ctxt *Link) {
	if Debug['v'] != 0 {
		fmt.Fprintf(ctxt.Bso, "%5.2f deadcode\n", obj.Cputime())
	}

	d := &deadcodepass{
		ctxt:        ctxt,
		ifaceMethod: make(map[methodsig]bool),
	}

	// First, flood fill any symbols directly reachable in the call
	// graph from INITENTRY. Ignore all methods not directly called.
	d.init()
	d.flood()

	callSym := Linkrlookup(ctxt, "reflect.Value.Call", 0)
	callSymSeen := false

	for {
		if callSym != nil && callSym.Attr.Reachable() {
			// Methods are called via reflection. Give up on
			// static analysis, mark all exported methods of
			// all reachable types as reachable.
			callSymSeen = true
		}

		// Mark all methods that could satisfy a discovered
		// interface as reachable. We recheck old marked interfaces
		// as new types (with new methods) may have been discovered
		// in the last pass.
		var rem []methodref
		for _, m := range d.markableMethods {
			if (callSymSeen && m.isExported()) || d.ifaceMethod[m.m] {
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

	// Remove all remaining unreached R_METHOD relocations.
	for _, m := range d.markableMethods {
		d.cleanupReloc(m.r0)
		d.cleanupReloc(m.r1)
	}

	if Buildmode != BuildmodeShared {
		// Keep a typelink if the symbol it points at is being kept.
		// (When BuildmodeShared, always keep typelinks.)
		for _, s := range ctxt.Allsym {
			if strings.HasPrefix(s.Name, "go.typelink.") {
				s.Attr.Set(AttrReachable, len(s.R) == 1 && s.R[0].Sym.Attr.Reachable())
			}
		}
	}

	// Remove dead text but keep file information (z symbols).
	var last *LSym
	for s := ctxt.Textp; s != nil; s = s.Next {
		if !s.Attr.Reachable() {
			continue
		}
		if last == nil {
			ctxt.Textp = s
		} else {
			last.Next = s
		}
		last = s
	}
	if last == nil {
		ctxt.Textp = nil
		ctxt.Etextp = nil
	} else {
		last.Next = nil
		ctxt.Etextp = last
	}
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
// method. There are two relocations, one for the method type without
// receiver, one with receiver
type methodref struct {
	m   methodsig
	src *LSym // receiver type symbol
	r0  *Reloc
	r1  *Reloc
}

func (m methodref) isExported() bool {
	for _, r := range m.m {
		return unicode.IsUpper(r)
	}
	panic("methodref has no signature")
}

// deadcodepass holds state for the deadcode flood fill.
type deadcodepass struct {
	ctxt            *Link
	markQueue       []*LSym            // symbols to flood fill in next pass
	ifaceMethod     map[methodsig]bool // methods declared in reached interfaces
	markableMethods []methodref        // methods of reached types
}

func (d *deadcodepass) cleanupReloc(r *Reloc) {
	if r.Sym.Attr.Reachable() {
		r.Type = obj.R_ADDR
	} else {
		if Debug['v'] > 1 {
			fmt.Fprintf(d.ctxt.Bso, "removing method %s\n", r.Sym.Name)
		}
		r.Sym = nil
		r.Siz = 0
	}
}

// mark appends a symbol to the mark queue for flood filling.
func (d *deadcodepass) mark(s, parent *LSym) {
	if s == nil || s.Attr.Reachable() {
		return
	}
	s.Attr |= AttrReachable
	s.Reachparent = parent
	d.markQueue = append(d.markQueue, s)
}

// markMethod marks a method as reachable and preps its R_METHOD relocations.
func (d *deadcodepass) markMethod(m methodref) {
	d.mark(m.r0.Sym, m.src)
	d.mark(m.r1.Sym, m.src)
	m.r0.Type = obj.R_ADDR
	m.r1.Type = obj.R_ADDR
}

// init marks all initial symbols as reachable.
// In a typical binary, this is INITENTRY.
func (d *deadcodepass) init() {
	var names []string

	if Thearch.Thechar == '5' {
		// mark some functions that are only referenced after linker code editing
		if d.ctxt.Goarm == 5 {
			names = append(names, "_sfloat")
		}
		names = append(names, "runtime.read_tls_fallback")
	}

	if Buildmode == BuildmodeShared {
		// Mark all symbols defined in this library as reachable when
		// building a shared library.
		for _, s := range d.ctxt.Allsym {
			if s.Type != 0 && s.Type != obj.SDYNIMPORT {
				d.mark(s, nil)
			}
		}
	} else {
		// In a normal binary, start at main.main and the init
		// functions and mark what is reachable from there.
		names = append(names, INITENTRY)
		if Linkshared && Buildmode == BuildmodeExe {
			names = append(names, "main.main", "main.init")
		}
		for _, name := range markextra {
			names = append(names, name)
		}
		for _, s := range dynexp {
			d.mark(s, nil)
		}
	}

	for _, name := range names {
		d.mark(Linkrlookup(d.ctxt, name, 0), nil)
	}
}

// flood flood fills symbols reachable from the markQueue symbols.
// As it goes, it collects methodref and interface method declarations.
func (d *deadcodepass) flood() {
	for len(d.markQueue) > 0 {
		s := d.markQueue[0]
		d.markQueue = d.markQueue[1:]
		if s.Type == obj.STEXT {
			if Debug['v'] > 1 {
				fmt.Fprintf(d.ctxt.Bso, "marktext %s\n", s.Name)
			}
			for _, a := range s.Autom {
				d.mark(a.Gotype, s)
			}
		}

		if strings.HasPrefix(s.Name, "type.") && s.Name[5] != '.' {
			if decodetype_kind(s)&kindMask == kindInterface {
				for _, sig := range decodetype_ifacemethods(s) {
					if Debug['v'] > 1 {
						fmt.Fprintf(d.ctxt.Bso, "reached iface method: %s\n", sig)
					}
					d.ifaceMethod[sig] = true
				}
			}
		}

		var methods []methodref
		for i := 0; i < len(s.R); i++ {
			r := &s.R[i]
			if r.Sym == nil {
				continue
			}
			if r.Type != obj.R_METHOD {
				d.mark(r.Sym, s)
				continue
			}
			// Collect rtype pointers to methods for
			// later processing in deadcode.
			if len(methods) > 0 {
				mref := &methods[len(methods)-1]
				if mref.r1 == nil {
					mref.r1 = r
					continue
				}
			}
			methods = append(methods, methodref{src: s, r0: r})
		}
		if len(methods) > 0 {
			// Decode runtime type information for type methods
			// to help work out which methods can be called
			// dynamically via interfaces.
			methodsigs := decodetype_methods(s)
			if len(methods) != len(methodsigs) {
				panic(fmt.Sprintf("%q has %d method relocations for %d methods", s.Name, len(methods), len(methodsigs)))
			}
			for i, m := range methodsigs {
				name := string(m)
				name = name[:strings.Index(name, "(")]
				if !strings.HasSuffix(methods[i].r0.Sym.Name, name) {
					panic(fmt.Sprintf("%q relocation for %q does not match method %q", s.Name, methods[i].r0.Sym.Name, name))
				}
				methods[i].m = m
			}
			d.markableMethods = append(d.markableMethods, methods...)
		}

		if s.Pcln != nil {
			for i := 0; i < s.Pcln.Nfuncdata; i++ {
				d.mark(s.Pcln.Funcdata[i], s)
			}
		}
		d.mark(s.Gotype, s)
		d.mark(s.Sub, s)
		d.mark(s.Outer, s)
	}
}
