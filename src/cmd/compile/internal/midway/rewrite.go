// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"internal/buildcfg"
	"strings"
)

// "Midway" rewriting
//
// Go attempts to provide a package similar to the the "Highway" library
// for C++ (https://google.github.io/highway).  The library package is "simd"
// and defines vector types with unspecified widths that are bound to particular
// machine dependent types as late as program execution.  This is accomplished
// by rewriting code that depends on these types into code that references
// architecture-specific types, perhaps more than once, and if necessary
// dynamically choosing which version to execute based on hardware attributes.
//
// The rewriting takes place early in the compiler, after type checking but
// before conversion to "unified" IR.  To ensure that types are correctly set
// on the modified version of the code, type checking information is reset and
// the type checking phase is re-run.  The places some limits on the shape of
// the rewrites, but it also ensures that the rewritten code is well-formed.
//
// Rewritten code does not reference "archsimd" types directly, but instead
// references types in a "bridge" package that filters the available methods
// and adds a few more.  The package used relies on a builder/compiler hack;
// the compiler's type checker enforces export naming conventions, but the
// build system limits visibility to unrelated "internal" packages and can be
// modified to allow access in special cases (like this one).  This allows the
// rewritten code to reference types, functions, and methods that are not
// accessible otherwise.

type Rewriter struct {
	pkg      *types2.Package
	analyzer *Analyzer
	info     *types2.Info
	sizes    []int
}

func NewRewriter(pkg *types2.Package, info *types2.Info, analyzer *Analyzer, sizes []int) *Rewriter {
	return &Rewriter{
		pkg:      pkg,
		info:     info,
		analyzer: analyzer,
		sizes:    sizes,
	}
}

func (r *Rewriter) Rewrite(files []*syntax.File) {

	// First duplicate and specialize all dependent functions and variables.
	for _, fileAST := range files {

		var newDecls []syntax.Decl
		for _, k := range r.sizes {
			newDecls = r.generateForSize(fileAST, k, newDecls)
		}

		// Then replace original functions with dispatchers.
		r.generateDispatchers(fileAST)

		fileAST.DeclList = append(fileAST.DeclList, newDecls...)
	}
}

func (r *Rewriter) generateDispatchers(fileAST *syntax.File) {
	var newDecls []syntax.Decl

	for _, decl := range fileAST.DeclList {
		switch d := decl.(type) {
		case *syntax.FuncDecl:
			if d.Name == nil {
				newDecls = append(newDecls, d)
				continue
			}
			obj := r.info.Defs[d.Name]
			if !r.analyzer.dependentObj[obj] || r.analyzer.inSimd {
				newDecls = append(newDecls, d)
				continue
			}

			sig, ok := obj.Type().(*types2.Signature)
			if !ok {
				newDecls = append(newDecls, d)
				continue
			}

			if r.analyzer.HasDependentSignature(sig) {
				// Drop dependent signatures entirely
				continue
			}

			// Clean signature -> Replace body with dispatcher
			d.Body = r.createDispatcherBody(d, sig)
			newDecls = append(newDecls, d)

		case *syntax.VarDecl:
			// Filter specs conceptually based on dependents
			keep := false
			for _, name := range d.NameList {
				if !r.analyzer.dependentObj[r.info.Defs[name]] {
					keep = true
					break // Keep entire var decl if any name is clean, else drop
				}
			}
			if keep {
				newDecls = append(newDecls, d)
			}
		case *syntax.TypeDecl:
			if !r.analyzer.dependentObj[r.info.Defs[d.Name]] || r.analyzer.inSimd {
				newDecls = append(newDecls, d)
			}
		default:
			newDecls = append(newDecls, decl)
		}
	}

	fileAST.DeclList = newDecls

	if !r.analyzer.inSimd {
		// Inject an import to the bridge package (if not exists)
		hasArchSimd := false
		var simdImport *syntax.ImportDecl
		for _, decl := range fileAST.DeclList {
			if imp, ok := decl.(*syntax.ImportDecl); ok {
				if imp.Path.Value == `"`+archFullPkg+`"` {
					hasArchSimd = true
				}
				if imp.Path.Value == `"`+simdPkg+`"` {
					simdImport = imp
				}

			}
		}
		p := simdImport.Pos()
		if !hasArchSimd {
			r.injectImport(fileAST, archFullPkg, p)
		}

		// Ensure at least one use of "simd"
		// var _ = simd.VectorBitLen()
		fun := &syntax.SelectorExpr{
			X:   syntax.NewName(p, simdPkg), // Assume this is resolvable
			Sel: syntax.NewName(p, vectorSizeFn),
		}
		fun.SetPos(p)
		call := &syntax.CallExpr{Fun: fun}
		call.SetPos(p)

		name := syntax.NewName(p, "_")

		varDecl := &syntax.VarDecl{NameList: []*syntax.Name{name}, Values: call}
		varDecl.SetPos(p)
		fileAST.DeclList = append(fileAST.DeclList, varDecl)
	}
}

func (r *Rewriter) injectImport(fileAST *syntax.File, toImport string, simdImportPos syntax.Pos) {
	importDecl := &syntax.ImportDecl{
		Path: &syntax.BasicLit{Value: `"` + toImport + `"`, Kind: syntax.StringLit},
	}
	importDecl.Path.SetPos(simdImportPos)
	importDecl.SetPos(simdImportPos)
	fileAST.DeclList = append([]syntax.Decl{importDecl}, fileAST.DeclList...)
}

func (r *Rewriter) createDispatcherBody(d *syntax.FuncDecl, sig *types2.Signature) *syntax.BlockStmt {

	// Build call arguments from the function parameters
	args := func() []syntax.Expr {
		var args []syntax.Expr
		if d.Type.ParamList != nil {
			for _, field := range d.Type.ParamList {
				if field.Name != nil {
					paramName := syntax.NewName(field.Pos(), field.Name.Value)
					args = append(args, paramName)
				}
			}
		}
		return args
	}

	// Slap a pos on an expression
	pe := func(e syntax.Expr) syntax.Expr {
		e.SetPos(d.Pos())
		return e
	}
	// Slap a pos on a statement
	ps := func(e syntax.Stmt) syntax.Stmt {
		e.SetPos(d.Pos())
		return e
	}

	// switch ast node.
	// the goal is something like (for now, till there are finer-grained choices)
	// switch simd.VectorSize() {
	//   case 128: if simd.Emulated() { call the specialize-for-emulation-code(args) }
	//             else { call the specialize-for-128-code(args) }
	//   case 256: call the specialize-for-256-code(args)
	//   etc
	// }
	//
	// the cases above deal with the usual `return call(...)` vs `call(...); return`
	switchStmt := &syntax.SwitchStmt{
		Tag: pe(&syntax.CallExpr{
			Fun: pe(&syntax.SelectorExpr{
				X:   syntax.NewName(d.Pos(), simdPkg), // Assume this is resolvable
				Sel: syntax.NewName(d.Pos(), vectorSizeFn),
			}),
		}),
		Body: []*syntax.CaseClause{},
	}

	var emulation syntax.Stmt

	for _, k := range r.sizes {
		fnName := fmt.Sprintf("%s@simd%d", d.Name.Value, k)
		fnIdent := syntax.NewName(d.Pos(), fnName)

		callExpr := pe(&syntax.CallExpr{
			Fun:     pe(fnIdent),
			ArgList: args(),
		})

		// callReturnStmt is either `return call(...)` or `call(...); return`
		var callReturnStmt syntax.Stmt
		if d.Type.ResultList != nil && len(d.Type.ResultList) > 0 {
			callReturnStmt = &syntax.ReturnStmt{Results: callExpr}
		} else {
			callReturnStmt = &syntax.BlockStmt{
				List: []syntax.Stmt{
					ps(&syntax.ExprStmt{X: callExpr}),
					ps(&syntax.ReturnStmt{}),
				},
				Rbrace: d.Pos(),
			}
		}
		callReturnStmt.SetPos(d.Pos())

		if k == 0 {
			// emulation == `if simd.Emulated() { callReturnStmt }`
			// save it for the first part of the 128 case.
			cond := pe(&syntax.CallExpr{
				Fun: pe(&syntax.SelectorExpr{
					X:   syntax.NewName(d.Pos(), simdPkg), // Assume this is resolvable
					Sel: syntax.NewName(d.Pos(), emulatedFn),
				})})

			blockStmt, ok := callReturnStmt.(*syntax.BlockStmt)
			if !ok {
				blockStmt = &syntax.BlockStmt{
					List:   []syntax.Stmt{callReturnStmt},
					Rbrace: d.Pos(),
				}
				blockStmt.SetPos(d.Pos())
			}

			emulation = ps(&syntax.IfStmt{
				Cond: cond,
				Then: blockStmt,
			})
			continue
		}

		var caseBody []syntax.Stmt
		// assume that 128 is a case; when we do scalable simd, this may change.
		// For now, if there is emulation, it is 128-bit (only).
		if emulation != nil && k == 128 {
			caseBody = append(caseBody, emulation)
			emulation = nil
		}

		caseClause := &syntax.CaseClause{
			Cases: pe(&syntax.BasicLit{Kind: syntax.IntLit, Value: fmt.Sprintf("%d", k)}),
			Body:  append(caseBody, callReturnStmt),
		}
		caseClause.SetPos(d.Pos())
		switchStmt.Body = append(switchStmt.Body, caseClause)
	}

	fnName := "panic"
	fnIdent := pe(syntax.NewName(d.Pos(), fnName))

	callExpr := pe(&syntax.CallExpr{
		Fun:     fnIdent,
		ArgList: []syntax.Expr{pe(&syntax.BasicLit{Value: "\"unsupported vector size in simd-rewritten code\"", Kind: syntax.StringLit})},
	})

	panicStmt := &syntax.ExprStmt{X: callExpr}
	blockStmt := &syntax.BlockStmt{List: []syntax.Stmt{ps(switchStmt), ps(panicStmt)}}

	blockStmt.SetPos(d.Pos())

	return blockStmt
}

func (r *Rewriter) generateForSize(fileAST *syntax.File, k int, newDecls []syntax.Decl) []syntax.Decl {
	copier := NewDeepCopier(r.pkg, r.info, k, r.analyzer, fmt.Sprintf("@simd%d", k))
	for _, decl := range fileAST.DeclList {
		if r.shouldIncludeDecl(decl) {
			newDecl := copier.CopyDecl(decl)
			newDecls = append(newDecls, newDecl)
		}
	}
	return newDecls
}

func nameToElemBitWidth(name string) int {
	var width int
	switch name {
	case "Int8s", "Uint8s", "Mask8s":
		width = 8
	case "Int16s", "Uint16s", "Mask16s":
		width = 16
	case "Int32s", "Uint32s", "Float32s", "Mask32s":
		width = 32
	case "Int64s", "Uint64s", "Float64s", "Mask64s":
		width = 64
	}
	return width
}

func (r *Rewriter) shouldIncludeDecl(decl syntax.Decl) bool {
	// Files (and declarations) in the simd package are excluded
	// from processing, except for those that whose name begins
	// with "tofrom_".
	if r.analyzer.inSimd {
		theFile := decl.Pos().Base().Filename()

		lastSlash := strings.LastIndex(theFile, simdPkg+"/")
		lastBackslash := strings.LastIndex(theFile, simdPkg+"\\")

		// Windows paths can be chaos, all we care, is whether the very last part
		// of the path is any-path-separator + "tofrom_" + anything-else, given that
		// we already know that we are in the simd package.
		maxSlash := max(lastSlash, lastBackslash)
		if maxSlash == -1 {
			return false
		}
		if !strings.HasPrefix(theFile[maxSlash:], simdPkg+"/tofrom_") &&
			!strings.HasPrefix(theFile[maxSlash:], simdPkg+"\\tofrom_") {
			return false
		}
	}

	switch d := decl.(type) {
	case *syntax.FuncDecl:
		if d.Name != nil {
			return r.analyzer.dependentObj[r.info.Defs[d.Name]]
		}
	case *syntax.TypeDecl:
		return r.analyzer.dependentObj[r.info.Defs[d.Name]]
	case *syntax.VarDecl:
		for _, name := range d.NameList {
			if r.analyzer.dependentObj[r.info.Defs[name]] {
				return true
			}
		}
	}
	return false
}

// Generate an API matching the standalone compilation call
func RewriteWrapper(pkg *types2.Package, info *types2.Info, files []*syntax.File) bool {
	if !buildcfg.Experiment.SIMD {
		return false
	}

	switch buildcfg.GOARCH {
	case "wasm", "amd64", "arm64":
	default:
		return false
	}

	sizes := rewriteSizes()
	if len(sizes) == 0 {
		return false
	}
	analyzer := NewAnalyzer(pkg, info)
	if !analyzer.Analyze(files) {
		return false
	}

	CheckPositions(files, "before midway")

	rewriter := NewRewriter(pkg, info, analyzer, sizes)
	rewriter.Rewrite(files)

	CheckPositions(files, "after midway")

	return true
}
