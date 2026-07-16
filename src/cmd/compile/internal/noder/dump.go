// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// MatchASTDump returns true if the fn matches the value
// of the astdump debug flag.
func MatchASTDump(fn *syntax.FuncDecl) bool {
	if len(base.Debug.AstDump) == 0 {
		return false
	}
	if fn.Name == nil {
		return false
	}
	return matchForDump(fn, base.Ctxt.Pkgpath)
}

// matchForDump is marked noinline to ensure that the exported
// function MatchAstDump IS inlineable and is also small, because
// common case is AstDump is not set.
//
//go:noinline
func matchForDump(fn *syntax.FuncDecl, pkgPath string) bool {
	return ir.MatchPkgFn(pkgPath, fn.Name.Value, base.Debug.AstDump)
}

func escapedFileName(fn *syntax.FuncDecl, suffix string) string {
	return ir.EscapedFileName(base.Ctxt.Pkgpath+"."+fn.Name.Value, suffix)
}

var mu sync.Mutex
var htmlWriters = make(map[*syntax.FuncDecl]*HTMLWriter)
var orderedFuncs = []*syntax.FuncDecl{}

// DumpNodeHTML dumps the node n to the HTML writer for fn.
func DumpNodeHTML(pkg *types2.Package, file *syntax.File, info *types2.Info, fn *syntax.FuncDecl, why string, n syntax.Node) {
	mu.Lock()
	defer mu.Unlock()
	w, ok := htmlWriters[fn]
	if !ok {
		name := escapedFileName(fn, ".syntax.html")
		w = NewHTMLWriter(pkg, file, info, name, fn, "")
		htmlWriters[fn] = w
		orderedFuncs = append(orderedFuncs, fn)
	}
	w.WritePhase(why, why)
}

// CloseHTMLWriters closes the HTML writer for fn, if one exists.
func CloseHTMLWriters() {
	mu.Lock()
	defer mu.Unlock()
	for _, fn := range orderedFuncs {
		if w, ok := htmlWriters[fn]; ok {
			w.Close("Writing html syntax output for %s to %s\n", w.pkgFuncName(), w.Path())
			delete(htmlWriters, fn)
		}
	}
	orderedFuncs = nil
}
