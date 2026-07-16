// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"internal/buildcfg"
	"strings"
)

// "Midway" rewriting
//
// Go attempts to provide a package similar to the "Highway" library
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
// the type checking phase is re-run.  This places some limits on the shape of
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
//
// The rewrite works in phases.  The first is "analysis", to discover functions,
// types, methods, and variables that depend on "simd" types.  "Depend on" means
// any mention of a simd type, and for types, also includes types that have a
// simd-dependent method.  Dependent functions are split into two categories;
// those whose dependence includes their signature, and those that do not.
// The second category forms the boundary between code that depends on simd and
// code that does not.  Notice that there cannot be a boundary method, because
// (by design) the receiver type is simd-dependent and thus a dependent method
// also has a dependent type in its signature.
//
// The second phase rewrites such "boundary" functions into a "dispatch" version
// and (later, third phase) "specialized" versions.  The dispatch function
// will choose which specialized version to call based on which simd implementation
// has been chosen, and forward parameters and results to/from that specialized version
// of the function.  The dispatch version shares the same name as the original function.
// Note that this applies to functions only, and not methods.

// The third phase specializes dependent functions (both kinds), methods,
// global variables, and types into size/emulation/feature-specific variants.
// Except for methods, this is done by adding a suffix beginning with "@" to
// the name.  Because "@" cannot appear in legal Go identifiers this removes
// the risk of a naming overlap.  Methods are specialized, but not renamed,
// because their receiver type is renamed instead.  Not changing method names
// preserves interface satisfaction, for example in the case of generic interfaces.
//
// Non-boundary dependent function and methods are not rewritten into dispatch
// functions/methods, but remain in the generated code because they must be
// present in the export data so that other packages that import them will still
// compile before rewriting.  Their bodies are replaced with panic(...) to allow
// compilation while preventing even worse chaos in the event of a bug either in
// the compiler or through ambitious use of reflection or assembly language.
//

/* Example rewrites

// Type alias, global variable, and init function:

// before:
type MyInt8s = simd.Int8s
func Generic[T haslen](x int) int {
    var v T
    return x + v.Len()
}
var VL int
func init() {
    VL = Generic[MyInt8s](1)
}
// dispatch:
func init() {
    switch simd.VectorBitSize() {
    case
        128:
            init@simd128()
            return
    case 256:
            init@simd256()
            return
    case 512:
            init@simd512()
            return
    default:
        panic("unsupported vector size")
    }
}
// specialized (128)
type MyInt8s@simd128 = archsimd.Int8x16
func init@simd128() {
        VL = Generic[MyInt8s@simd128](1)
}


// structure containing simd fields, and with simd methods

// before
// A struct dependent on SIMD
type VectorC struct {
    Field simd.Float32s
}
func (v *VectorC) MethodOfSimd() bool {
    return false
}
func (v VectorC) Data() simd.Float32s {
    return v.Field
}
func (v VectorC) Foo(x VectorC) VectorC {
    return VectorC{Field: v.Field.Add(x.Field)}
}

// dispatch
// technically there is none, but functions with panicking bodies
// remain because code must pass type checking before rewriting.
type VectorC struct {
    Field simd.Float32s
}
func (v *VectorC) MethodOfSimd() bool {
    panic(...)
}
func (v VectorC) Data() simd.Float32s {
    panic(...)
}
func (v VectorC) Foo(x VectorC) VectorC {
    panic(...)
}

// specialized (128)

// A struct dependent on SIMD
type VectorC@simd128 struct {
    Field bridge.Float32x4
}
func (v *VectorC@simd128) MethodOfSimd() bool {
    return false
}
func (v VectorC@simd128) Data() bridge.Float32x4 {
    return v.Field
}
func (v VectorC@simd128) Foo(x VectorC@simd128) VectorC@simd128 {
    return VectorC@simd128{Field: v.Field.Add(x.Field)}
}

*/

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
		// This also edits the DeclList of fileAST.
		r.generateDispatchers(fileAST)

		fileAST.DeclList = append(fileAST.DeclList, newDecls...)
	}
}

func (r *Rewriter) generateDispatchers(fileAST *syntax.File) {
	var newDecls []syntax.Decl

	change := false

	for _, decl := range fileAST.DeclList {
		switch d := decl.(type) {
		case *syntax.FuncDecl:
			if d.Name == nil {
				newDecls = append(newDecls, d)
				continue
			}
			obj := r.info.Defs[d.Name]
			if !r.analyzer.isDependentObj[obj] || r.analyzer.inSimd {
				newDecls = append(newDecls, d)
				continue
			}

			sig, ok := obj.Type().(*types2.Signature)
			if !ok {
				newDecls = append(newDecls, d)
				continue
			}

			change = true
			if r.analyzer.HasDependentSignature(sig) {
				if base.Debug.Simd > 0 {
					base.Warn("%s: removing body of dependent-sig original function %v", d.Pos().String(), d.Name.Value)
				}
				d.Body = r.blockOf(d.Pos(), r.panicStmt(d.Pos(),
					"unexpected call of original function rewritten to specialized SIMD"))
				newDecls = append(newDecls, d)
				continue
			}

			// Clean signature -> Replace body with dispatcher
			d.Body = r.createDispatcherBody(d, sig)
			newDecls = append(newDecls, d)

		case *syntax.VarDecl:
			// Keep var decls even if rewritten, so that pre-rewrite code parses correctly.
			// TODO figure out how to deal with side-effects in initializers.
			newDecls = append(newDecls, d)

		case *syntax.TypeDecl:
			// Keep all types; we need the untranslated copy if a method referencing it
			// needs to typecheck pre-translation.
			newDecls = append(newDecls, d)
		default:
			newDecls = append(newDecls, decl)
		}
	}

	if !change {
		return
	}

	fileAST.DeclList = newDecls

	if !r.analyzer.inSimd {
		// Inject an import to the bridge package (if not exists)
		hasArchSimd := false
		var simdImport *syntax.ImportDecl
		p := fileAST.Pos()
		for _, decl := range fileAST.DeclList {
			if imp, ok := decl.(*syntax.ImportDecl); ok {
				if imp.Path.Value == `"`+archFullPkg+`"` {
					hasArchSimd = true
					if simdImport == nil {
						p = imp.Pos()
					}
				}
				if imp.Path.Value == `"`+simdPkg+`"` {
					simdImport = imp
					p = imp.Pos()
				}
			}
		}

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

	panicStmt := r.panicStmt(d.Pos(), "unsupported vector size in simd-rewritten code")
	return r.blockOf(d.Pos(), switchStmt, panicStmt)
}

func (r *Rewriter) blockOf(p syntax.Pos, stmts ...syntax.Stmt) *syntax.BlockStmt {
	for _, s := range stmts {
		s.SetPos(p)
	}
	blockStmt := &syntax.BlockStmt{List: stmts}
	blockStmt.SetPos(p)
	return blockStmt
}

func (r *Rewriter) panicStmt(p syntax.Pos, unquotedMessage string) *syntax.ExprStmt {
	pe := func(e syntax.Expr) syntax.Expr {
		e.SetPos(p)
		return e
	}
	fnName := "panic"
	fnIdent := pe(syntax.NewName(p, fnName))
	callExpr := pe(&syntax.CallExpr{
		Fun:     fnIdent,
		ArgList: []syntax.Expr{pe(&syntax.BasicLit{Value: `"` + unquotedMessage + `"`, Kind: syntax.StringLit})},
	})
	panicStmt := &syntax.ExprStmt{X: callExpr}
	panicStmt.SetPos(p)
	return panicStmt
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
			return r.analyzer.isDependentObj[r.info.Defs[d.Name]]
		}
	case *syntax.TypeDecl:
		return r.analyzer.isDependentObj[r.info.Defs[d.Name]]
	case *syntax.VarDecl:
		for _, name := range d.NameList {
			if r.analyzer.isDependentObj[r.info.Defs[name]] {
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
