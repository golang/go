// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ld

import (
	"cmd/internal/obj"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"os"
	"sync"
)

type unresolvedSymKey struct {
	from *sym.Symbol // Symbol that referenced unresolved "to"
	to   *sym.Symbol // Unresolved symbol referenced by "from"
}

type lookupFn func(name string, version int) *sym.Symbol
type symNameFn func(s loader.Sym) string

// ErrorReporter is used to make error reporting thread safe.
type ErrorReporter struct {
	unresOnce  sync.Once
	unresSyms  map[unresolvedSymKey]bool
	unresMutex sync.Mutex
	lookup     lookupFn
	SymName    symNameFn
}

// errorUnresolved prints unresolved symbol error for r.Sym that is referenced from s.
func (reporter *ErrorReporter) errorUnresolved(s *sym.Symbol, r *sym.Reloc) {
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

// Errorf method logs an error message.
//
// If more than 20 errors have been printed, exit with an error.
//
// Logging an error means that on exit cmd/link will delete any
// output file and return a non-zero error code.
// TODO: consolidate the various different versions of Errorf (
// function, Link method, and ErrorReporter method).
func (reporter *ErrorReporter) Errorf(s loader.Sym, format string, args ...interface{}) {
	if s != 0 && reporter.SymName != nil {
		sn := reporter.SymName(s)
		format = sn + ": " + format
	} else {
		format = fmt.Sprintf("sym %d: %s", s, format)
	}
	format += "\n"
	fmt.Fprintf(os.Stderr, format, args...)
	afterErrorAction()
}
