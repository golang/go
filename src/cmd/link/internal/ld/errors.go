// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ld

import (
	"cmd/internal/obj"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"sync"
)

type unresolvedSymKey struct {
	from loader.Sym // Symbol that referenced unresolved "to"
	to   loader.Sym // Unresolved symbol referenced by "from"
}

type unresolvedSymKey2 struct {
	from *sym.Symbol // Symbol that referenced unresolved "to"
	to   *sym.Symbol // Unresolved symbol referenced by "from"
}

type lookupFn func(name string, version int) *sym.Symbol
type symNameFn func(s loader.Sym) string

// ErrorReporter is used to make error reporting thread safe.
type ErrorReporter struct {
	loader.ErrorReporter
	unresOnce  sync.Once
	unresSyms  map[unresolvedSymKey]bool
	unresSyms2 map[unresolvedSymKey2]bool
	unresMutex sync.Mutex
	lookup     lookupFn
	SymName    symNameFn
}

// errorUnresolved prints unresolved symbol error for rs that is referenced from s.
func (reporter *ErrorReporter) errorUnresolved(ldr *loader.Loader, s, rs loader.Sym) {
	reporter.unresOnce.Do(func() { reporter.unresSyms = make(map[unresolvedSymKey]bool) })

	k := unresolvedSymKey{from: s, to: rs}
	reporter.unresMutex.Lock()
	defer reporter.unresMutex.Unlock()
	if !reporter.unresSyms[k] {
		reporter.unresSyms[k] = true
		name := ldr.SymName(rs)

		// Try to find symbol under another ABI.
		var reqABI, haveABI obj.ABI
		haveABI = ^obj.ABI(0)
		reqABI, ok := sym.VersionToABI(ldr.SymVersion(rs))
		if ok {
			for abi := obj.ABI(0); abi < obj.ABICount; abi++ {
				v := sym.ABIToVersion(abi)
				if v == -1 {
					continue
				}
				if rs1 := ldr.Lookup(name, v); rs1 != 0 && ldr.SymType(rs1) != sym.Sxxx && ldr.SymType(rs1) != sym.SXREF {
					haveABI = abi
				}
			}
		}

		// Give a special error message for main symbol (see #24809).
		if name == "main.main" {
			reporter.Errorf(s, "function main is undeclared in the main package")
		} else if haveABI != ^obj.ABI(0) {
			reporter.Errorf(s, "relocation target %s not defined for %s (but is defined for %s)", name, reqABI, haveABI)
		} else {
			reporter.Errorf(s, "relocation target %s not defined", name)
		}
	}
}

// errorUnresolved2 prints unresolved symbol error for r.Sym that is referenced from s.
func (reporter *ErrorReporter) errorUnresolved2(s *sym.Symbol, r *sym.Reloc) {
	reporter.unresOnce.Do(func() { reporter.unresSyms2 = make(map[unresolvedSymKey2]bool) })

	k := unresolvedSymKey2{from: s, to: r.Sym}
	reporter.unresMutex.Lock()
	defer reporter.unresMutex.Unlock()
	if !reporter.unresSyms2[k] {
		reporter.unresSyms2[k] = true

		// Try to find symbol under another ABI.
		var reqABI, haveABI obj.ABI
		haveABI = ^obj.ABI(0)
		reqABI, ok := sym.VersionToABI(int(r.Sym.Version))
		if ok {
			for abi := obj.ABI(0); abi < obj.ABICount; abi++ {
				v := sym.ABIToVersion(abi)
				if v == -1 {
					continue
				}
				if rs := reporter.lookup(r.Sym.Name, v); rs != nil && rs.Type != sym.Sxxx && rs.Type != sym.SXREF {
					haveABI = abi
				}
			}
		}

		// Give a special error message for main symbol (see #24809).
		if r.Sym.Name == "main.main" {
			Errorf(s, "function main is undeclared in the main package")
		} else if haveABI != ^obj.ABI(0) {
			Errorf(s, "relocation target %s not defined for %s (but is defined for %s)", r.Sym.Name, reqABI, haveABI)
		} else {
			Errorf(s, "relocation target %s not defined", r.Sym.Name)
		}
	}
}
