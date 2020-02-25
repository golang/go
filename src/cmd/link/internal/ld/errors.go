// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ld

import (
	"cmd/internal/obj"
	"cmd/link/internal/sym"
	"sync"
)

type unresolvedSymKey struct {
	from *sym.Symbol // Symbol that referenced unresolved "to"
	to   *sym.Symbol // Unresolved symbol referenced by "from"
}

// ErrorReporter is used to make error reporting thread safe.
type ErrorReporter struct {
	unresOnce  sync.Once
	unresSyms  map[unresolvedSymKey]bool
	unresMutex sync.Mutex
}

type roLookup func(name string, v int) *sym.Symbol

// errorUnresolved prints unresolved symbol error for r.Sym that is referenced from s.
func (reporter *ErrorReporter) errorUnresolved(lookup roLookup, s *sym.Symbol, r *sym.Reloc) {
	reporter.unresOnce.Do(func() { reporter.unresSyms = make(map[unresolvedSymKey]bool) })

	k := unresolvedSymKey{from: s, to: r.Sym}
	reporter.unresMutex.Lock()
	defer reporter.unresMutex.Unlock()
	if !reporter.unresSyms[k] {
		reporter.unresSyms[k] = true

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
				if rs := lookup(r.Sym.Name, v); rs != nil && rs.Type != sym.Sxxx {
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
