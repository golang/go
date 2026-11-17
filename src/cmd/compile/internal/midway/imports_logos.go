// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Source identifiers cannot spell these compiler-only import bindings.
const simdRewriteName = "simd@midway"
const bridgeRewriteName = "bridge@midway"

func (c *DeepCopier) bridgeQualifier() string {
	if c.analyzer.inSimd {
		return archPkg
	}
	return bridgeRewriteName
}

// injectRewriteImports runs after rewriting one already-checked file. Generated
// selectors need their own bindings even when only an imported type is SIMD
// dependent, or user declarations shadow the usual package names.
func (r *Rewriter) injectRewriteImports(file *syntax.File) {
	var uses []syntax.Decl
	for _, decl := range file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok {
			continue
		}
		obj := r.info.Implicits[imp]
		if imp.LocalPkgName != nil {
			obj = r.info.Defs[imp.LocalPkgName]
		}
		pkg := obj.(*types2.PkgName)
		if use := rewriteImportUse(imp.Pos(), pkg.Name(), pkg.Imported().Path()); use != nil {
			uses = append(uses, use)
		}
	}
	p := file.Pos()
	var imports []syntax.Decl
	for _, binding := range []struct{ path, name string }{
		{simdPkg, simdRewriteName},
		{archFullPkg, bridgeRewriteName},
	} {
		imp := &syntax.ImportDecl{
			LocalPkgName: syntax.NewName(p, binding.name),
			Path:         &syntax.BasicLit{Value: `"` + binding.path + `"`, Kind: syntax.StringLit},
		}
		imp.SetPos(p)
		imp.Path.SetPos(p)
		imports = append(imports, imp)
		uses = append(uses, rewriteImportUse(p, binding.name, binding.path))
	}
	file.DeclList = append(append(imports, file.DeclList...), uses...)
}

// rewriteImportUse preserves import use without evaluating a runtime call.
// Initial source checking has already rejected genuinely unused source imports.
func rewriteImportUse(p syntax.Pos, qualifier, path string) syntax.Decl {
	if qualifier == "_" || path != simdPkg && path != archFullPkg {
		return nil
	}
	member := vectorSizeFn
	if path == archFullPkg {
		member = "ZeroSized"
	}
	memberName := syntax.NewName(p, member)
	var value syntax.Expr = memberName
	if qualifier != "." {
		selector := &syntax.SelectorExpr{X: syntax.NewName(p, qualifier), Sel: memberName}
		selector.SetPos(p)
		value = selector
	}
	decl := &syntax.VarDecl{NameList: []*syntax.Name{syntax.NewName(p, "_")}}
	decl.SetPos(p)
	if path == archFullPkg {
		decl.Type = value
	} else {
		decl.Values = value
	}
	return decl
}
