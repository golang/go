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
// global variables, and types into size- and emulation-specific versions.
// Except for methods, this is done by adding a suffix beginning with "@" to
// the name.  Because "@" cannot appear in legal Go identifiers this removes
// the risk of a naming overlap.  Methods are specialized, but not renamed,
// because their receiver type is renamed instead.  Not changing method names
// preserves interface satisfaction, for example in the case of generic interfaces.
//
// Non-boundary dependent function and methods are not rewritten into dispatch
// functions/methods, but remain in the generated code because they must be
// present in the export data so that other packages that import them will still
// compile before rewriting.  Their bodies are replaced with panic(...) so that
// they still type check.
//
// These, and the dependent types and variables, which also remain, are the
// "originals".  Rewritten code never refers to an original; it refers to the
// specialized versions instead.  In this respect an original is like a generic
// declaration whose type argument is the vector size, and the specialized
// versions are its instantiations.  As with a generic declaration, the compiler
// writes an original to the export data but doesn't compile it: no code, type
// descriptor, or storage is generated for it.  RewriteWrapper reports which
// declarations are originals.
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
    switch simd.VectorBits() {
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
	pkg          *types2.Package
	analyzer     *Analyzer
	info         *types2.Info
	sizes        []int
	blankCounter int
}

func NewRewriter(pkg *types2.Package, info *types2.Info, analyzer *Analyzer, sizes []int) *Rewriter {
	return &Rewriter{
		pkg:      pkg,
		info:     info,
		analyzer: analyzer,
		sizes:    sizes,
	}
}

func (r *Rewriter) newCopier(k int) *DeepCopier {
	return NewDeepCopier(r.pkg, r.info, k, r.analyzer, fmt.Sprintf("%s%d", SimdMark, k))
}

func (r *Rewriter) nextBlankTarget() string {
	id := r.blankCounter
	r.blankCounter++
	return fmt.Sprintf("_@%d", id)
}

func (r *Rewriter) Rewrite(files []*syntax.File) {
	// First duplicate and specialize all dependent functions and types,
	// collecting them in newDecls to append at the end of the file.
	// Top-level variable declarations are instead rewritten in place during
	// generateDispatchers so that source declaration order (and thus
	// types2 initialization order) is preserved.
	for _, fileAST := range files {

		var newDecls []syntax.Decl
		for _, k := range r.sizes {
			newDecls = r.generateForSize(fileAST, k, newDecls)
		}

		// Then replace original functions with dispatchers and rewrite
		// top-level variable declarations in place.
		// This also edits the DeclList of fileAST.
		r.generateDispatchers(fileAST)

		fileAST.DeclList = append(fileAST.DeclList, newDecls...)
	}
}

func (r *Rewriter) generateDispatchers(fileAST *syntax.File) {
	var newDecls []syntax.Decl

	declListChanged := false

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

			declListChanged = true
			if r.analyzer.HasDependentSignature(sig) {
				if base.Debug.Simd > 0 {
					base.Warn("%s: removing body of dependent-sig original function %v", d.Pos().String(), d.Name.Value)
				}
				d.Body = r.blockOf(d.Pos(), r.panicStmt(d.Pos(),
					"unexpected call of original function rewritten to specialized SIMD"))
				newDecls = append(newDecls, d)
				continue
			}

			// Clean signature -> Replace body with dispatcher.
			// For methods, ensure the receiver has a usable name so the dispatcher
			// can forward the call to the specialized method on the receiver.
			if d.Recv != nil && (d.Recv.Name == nil || d.Recv.Name.Value == "_") {
				d.Recv.Name = syntax.NewName(d.Recv.Pos(), "recv@")
			}
			d.Body = r.createDispatcherBody(d, sig)
			newDecls = append(newDecls, d)

		case *syntax.VarDecl:
			rewritten, changed := r.rewriteTopLevelVarDecl(d)
			if changed {
				declListChanged = true
			}
			newDecls = append(newDecls, rewritten...)

		case *syntax.TypeDecl:
			// Mark changed even for type-only files so @bridge is imported for
			// the specialized type definitions appended by generateForSize.
			if r.shouldIncludeDecl(d) && !r.analyzer.inSimd {
				declListChanged = true
			}
			// Keep all types; we need the untranslated copy if a method referencing it
			// needs to typecheck pre-translation.
			newDecls = append(newDecls, d)
		default:
			newDecls = append(newDecls, decl)
		}
	}

	if !declListChanged {
		return
	}

	fileAST.DeclList = newDecls

	if !r.analyzer.inSimd {
		// Inject private aliases for the simd and bridge packages so midway-generated
		// references never collide with user declarations, imports, or locals.
		var userSimdImports []*syntax.ImportDecl
		p := fileAST.Pos()
		for _, decl := range fileAST.DeclList {
			if imp, ok := decl.(*syntax.ImportDecl); ok {
				if imp.Path.Value == `"`+SimdBridgePkg+`"` {
					if len(userSimdImports) == 0 {
						p = imp.Pos()
					}
				}
				if imp.Path.Value == `"`+SimdPkg+`"` {
					if imp.LocalPkgName == nil || imp.LocalPkgName.Value != "_" {
						userSimdImports = append(userSimdImports, imp)
					}
					p = imp.Pos()
				}
			}
		}

		r.injectImport(fileAST, simdPkgAlias, SimdPkg, p)
		r.injectImport(fileAST, archPkgAlias, SimdBridgePkg, p)

		// Ensure at least one use of the injected @simd import and any user
		// imports of "simd" (unaliased, aliased, or dot-imported) whose
		// references may have been rewritten.
		// var _ = <pkgIdent>.VectorBits()  (or var _ = VectorBits() for dot-import)
		appendSimdUse := func(pkgIdent string, pos syntax.Pos) {
			var fun syntax.Expr
			if pkgIdent == "." {
				fun = syntax.NewName(pos, vectorSizeFn)
			} else {
				sel := &syntax.SelectorExpr{
					X:   syntax.NewName(pos, pkgIdent),
					Sel: syntax.NewName(pos, vectorSizeFn),
				}
				sel.SetPos(pos)
				fun = sel
			}
			call := &syntax.CallExpr{Fun: fun}
			call.SetPos(pos)
			name := syntax.NewName(pos, "_")
			varDecl := &syntax.VarDecl{NameList: []*syntax.Name{name}, Values: call}
			varDecl.SetPos(pos)
			fileAST.DeclList = append(fileAST.DeclList, varDecl)
		}
		appendSimdUse(simdPkgAlias, p)
		for _, imp := range userSimdImports {
			ident := SimdPkg
			if imp.LocalPkgName != nil {
				ident = imp.LocalPkgName.Value
			}
			appendSimdUse(ident, imp.Pos())
		}

		// Ensure at least one use of the injected @bridge import
		// var _ @bridge.ZeroSized
		bridgeType := &syntax.SelectorExpr{
			X:   syntax.NewName(p, archPkgAlias),
			Sel: syntax.NewName(p, "ZeroSized"),
		}
		bridgeType.SetPos(p)
		bridgeName := syntax.NewName(p, "_")
		bridgeVarDecl := &syntax.VarDecl{NameList: []*syntax.Name{bridgeName}, Type: bridgeType}
		bridgeVarDecl.SetPos(p)
		fileAST.DeclList = append(fileAST.DeclList, bridgeVarDecl)
	}
}

func (r *Rewriter) injectImport(fileAST *syntax.File, localName, toImport string, simdImportPos syntax.Pos) {
	importDecl := &syntax.ImportDecl{
		LocalPkgName: syntax.NewName(simdImportPos, localName),
		Path:         &syntax.BasicLit{Value: `"` + toImport + `"`, Kind: syntax.StringLit},
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
			for i, field := range d.Type.ParamList {
				if field.Name == nil || field.Name.Value == "_" {
					// Blank and unnamed parameters cannot be passed along, so name them.
					field.Name = syntax.NewName(field.Pos(), fmt.Sprintf("p@%d", i))
				}
				paramName := syntax.NewName(field.Pos(), field.Name.Value)
				args = append(args, paramName)
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

	simdCallExpr := func(fnName string) syntax.Expr {
		return pe(&syntax.CallExpr{
			Fun: pe(&syntax.SelectorExpr{
				X:   syntax.NewName(d.Pos(), simdPkgAlias),
				Sel: syntax.NewName(d.Pos(), fnName),
			}),
		})
	}

	// switch ast node.
	// the goal is something like (for now, till there are finer-grained choices)
	// switch @simd.VectorBits() {
	//   case 128: if @simd.Emulated() { call the specialize-for-emulation-code(args) }
	//             else { call the specialize-for-128-code(args) }
	//   case 256: call the specialize-for-256-code(args)
	//   etc
	// }
	//
	// Keep this runtime dispatch logic in sync with noder.(*reader).simdInitGuard.
	//
	// the cases above deal with the usual `return call(...)` vs `call(...); return`
	switchStmt := &syntax.SwitchStmt{
		Tag:  simdCallExpr(vectorSizeFn),
		Body: []*syntax.CaseClause{},
	}

	var emulation syntax.Stmt

	makeCallReturnStmt := func(k int) syntax.Stmt {
		fnName := fmt.Sprintf("%s%s%d", d.Name.Value, SimdMark, k)
		fnIdent := syntax.NewName(d.Pos(), fnName)

		var fun syntax.Expr
		if d.Recv != nil && d.Recv.Name != nil {
			// For methods, call the specialized method on the receiver rather
			// than as a plain function, so the type-checker can resolve it.
			recvIdent := syntax.NewName(d.Pos(), d.Recv.Name.Value)
			selExpr := &syntax.SelectorExpr{X: recvIdent, Sel: fnIdent}
			selExpr.SetPos(d.Pos())
			fun = selExpr
		} else {
			fun = pe(fnIdent)
		}

		callExpr := pe(&syntax.CallExpr{
			Fun:     fun,
			ArgList: args(),
			HasDots: sig.Variadic(),
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
		return callReturnStmt
	}

	guardCallWithCondition := func(require string, stmt syntax.Stmt) syntax.Stmt {
		cond := simdCallExpr(require)

		blockStmt, ok := stmt.(*syntax.BlockStmt)
		if !ok {
			blockStmt = &syntax.BlockStmt{
				List:   []syntax.Stmt{stmt},
				Rbrace: d.Pos(),
			}
			blockStmt.SetPos(d.Pos())
		}

		guarded := ps(&syntax.IfStmt{
			Cond: cond,
			Then: blockStmt,
		})
		return guarded
	}

	for _, k := range r.sizes {

		callReturnStmt := makeCallReturnStmt(k)

		if k == 0 {
			emulation = guardCallWithCondition(emulatedFn, callReturnStmt)
			continue
		}

		var caseBody []syntax.Stmt
		// assume that 128 is a case; when we do scalable simd, this may change.
		// For now, if there is emulation, it is 128-bit (only).
		if emulation != nil && k == 128 {
			caseBody = append(caseBody, emulation)
			emulation = nil
		}

		caseBody = append(caseBody, callReturnStmt)

		caseClause := &syntax.CaseClause{
			Cases: pe(&syntax.BasicLit{Kind: syntax.IntLit, Value: fmt.Sprintf("%d", k)}),
			Body:  caseBody,
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
	copier := r.newCopier(k)
	for _, decl := range fileAST.DeclList {
		if _, isVar := decl.(*syntax.VarDecl); isVar {
			// Top-level VarDecls are rewritten in place during generateDispatchers.
			continue
		}
		if r.shouldIncludeDecl(decl) {
			newDecl := copier.CopyDecl(decl)
			newDecls = append(newDecls, newDecl)
		}
	}
	return newDecls
}

// rewriteTopLevelVarDecl rewrites a package-level VarDecl in place.
//
// Variables whose type depends on SIMD are split into specialized copies
// (<name>@simd<size>) while retaining the original declaration for
// pre-rewrite typechecking and export data.
//
// Variables whose type does NOT depend on SIMD, but whose initializer depends
// on SIMD (including blank "_" initializers), remain a single variable: midway
// emits one synthetic init-proxy declaration (@simdinit<size>@<target>)
// per SIMD configuration immediately before the original variable, and rewrites
// the original non-blank variable's initializer to reference the last proxy so
// that types2 preserves its inferred type. After types2 re-checks the rewritten
// AST, InitOrderPlan.Apply drops the pre-rewrite dummy initializers and restores
// the exact pre-midway Go-spec initialization order with each declaration's
// specialized declarations grouped atomically. The unified IR writer omits proxy
// declarations from pkgDecl, and the unified IR reader rewrites each proxy
// initializer into a conditional assignment directly to the real variable (or
// blank).
func (r *Rewriter) rewriteTopLevelVarDecl(d *syntax.VarDecl) ([]syntax.Decl, bool) {
	// Package simd's own var initializers run before simd's init() sets
	// emulated/maxVectorSize, so SIMD-guarded package-var initializers cannot
	// work inside package simd itself; tofrom_*.go only needs FuncDecl
	// specialization.
	if r.analyzer.inSimd {
		return []syntax.Decl{d}, false
	}

	// Case 1: no initializer.
	//   var V simd.Float32s
	// =>
	//   var V@simd0 @bridge.Float32s
	//   var V@simd128 @bridge.Float32x4
	//   ...
	//   var V simd.Float32s // retained for export data / pre-rewrite refs
	if d.Values == nil {
		hasDepType := false
		for _, name := range d.NameList {
			if obj := r.info.Defs[name]; obj != nil && r.analyzer.isDependentObj[obj] {
				hasDepType = true
				break
			}
		}
		if !hasDepType {
			return []syntax.Decl{d}, false
		}
		var out []syntax.Decl
		for _, k := range r.sizes {
			out = append(out, r.newCopier(k).CopyVarDecl(d))
		}
		out = append(out, d)
		return out, true
	}

	// Case 2: parallel N:N declaration ("var a, b [T] = e1, e2").
	// Split into individual 1:1 declarations if any item depends on SIMD so
	// non-dependent siblings stay untouched and source order is preserved:
	//   var a, b = 30, simdExpr
	// =>
	//   var a = 30
	//   <Case 3 rewrite of: var b = simdExpr>
	if list, ok := d.Values.(*syntax.ListExpr); ok {
		var out []syntax.Decl
		anyChanged := false
		for i, name := range d.NameList {
			item := &syntax.VarDecl{
				Group:    d.Group,
				Pragma:   d.Pragma,
				NameList: []*syntax.Name{name},
				Type:     d.Type,
				Values:   list.ElemList[i],
			}
			item.SetPos(name.Pos())
			sub, changed := r.rewriteSingleVarDecl(item, i > 0)
			anyChanged = anyChanged || changed
			out = append(out, sub...)
		}
		if !anyChanged {
			return []syntax.Decl{d}, false
		}
		return out, true
	}

	// Case 3: single 1:1 declaration ("var x [T] = expr").
	// - SIMD-typed:
	//     var V = expr
	//   =>
	//     var V@simd<k> = expr<k> ...
	//     var V = expr // dummy init dropped from InitOrder
	// - Non-SIMD-typed ("boundary" var):
	//     var N = expr
	//   =>
	//     var @simdinit<k>@N = expr<k> ... // omitted by writer, redirected in reader
	//     var N = @simdinit<last>@N        // dummy init dropped from InitOrder
	if len(d.NameList) == 1 {
		return r.rewriteSingleVarDecl(d, false)
	}

	// Case 4: multi-variable N:1 declaration ("var a, b [T] = multiReturn()").
	// E.g. for non-SIMD `a` and SIMD-typed `b`:
	//   var a, b = pair()
	// =>
	//   var @simdinit<k>@a, b@simd<k> = pair@simd<k>() ...
	//   var a = @simdinit<last>@a // dummy init dropped from InitOrder
	//   var _, b = pair()         // retained for b's export data; dropped from InitOrder
	hasDepType := false
	for _, name := range d.NameList {
		if obj := r.info.Defs[name]; obj != nil && r.analyzer.isDependentObj[obj] {
			hasDepType = true
			break
		}
	}
	hasDepInit := r.analyzer.hasNodeDependency(d.Values) || (d.Type != nil && r.analyzer.hasNodeDependency(d.Type))
	// Non-SIMD multi-return var in a file that uses SIMD elsewhere.
	if !hasDepType && !hasDepInit {
		return []syntax.Decl{d}, false
	}

	blankTargets := make([]string, len(d.NameList))
	for i, name := range d.NameList {
		if name.Value == "_" {
			blankTargets[i] = r.nextBlankTarget()
		}
	}

	var out []syntax.Decl
	for _, k := range r.sizes {
		copier := r.newCopier(k)
		specNames := make([]*syntax.Name, len(d.NameList))
		for i, name := range d.NameList {
			obj := r.info.Defs[name]
			switch {
			case obj != nil && r.analyzer.isDependentObj[obj]:
				specNames[i] = copier.CopyName(name, true)
			case name.Value == "_":
				specNames[i] = syntax.NewName(name.Pos(), SimdInitProxyName(blankTargets[i], k))
			default:
				proxy := SimdInitProxyName(name.Value, k)
				if base.Debug.Simd > 0 {
					base.Warn("%s: creating simd init proxy %s for %s", name.Pos().String(), proxy, name.Value)
				}
				specNames[i] = syntax.NewName(name.Pos(), proxy)
			}
		}
		specDecl := &syntax.VarDecl{
			Group:    d.Group,
			NameList: specNames,
			Type:     copier.CopyExpr(d.Type),
			Values:   copier.CopyExpr(d.Values),
		}
		specDecl.SetPos(d.Pos())
		out = append(out, specDecl)
	}

	usedType := false
	nextOrigType := func() syntax.Expr {
		if d.Type == nil {
			return nil
		}
		if !usedType {
			usedType = true
			return d.Type
		}
		return cloneExpr(d.Type)
	}

	lastSize := r.sizes[len(r.sizes)-1]
	for _, name := range d.NameList {
		obj := r.info.Defs[name]
		if name.Value == "_" || (obj != nil && r.analyzer.isDependentObj[obj]) {
			continue
		}
		lastProxyRef := syntax.NewName(name.Pos(), SimdInitProxyName(name.Value, lastSize))
		origDecl := &syntax.VarDecl{
			Group:    d.Group,
			Pragma:   d.Pragma,
			NameList: []*syntax.Name{name},
			Type:     nextOrigType(),
			Values:   lastProxyRef,
		}
		origDecl.SetPos(name.Pos())
		out = append(out, origDecl)
	}

	if hasDepType {
		// Retain original declaration for the SIMD-typed variables (replacing
		// non-SIMD-typed variables with "_") so export data and pre-rewrite
		// references continue to see their original SIMD types.
		origNames := make([]*syntax.Name, len(d.NameList))
		for i, name := range d.NameList {
			if obj := r.info.Defs[name]; obj != nil && r.analyzer.isDependentObj[obj] {
				origNames[i] = name
			} else {
				origNames[i] = syntax.NewName(name.Pos(), "_")
			}
		}
		origSimd := &syntax.VarDecl{
			Group:    d.Group,
			Pragma:   d.Pragma,
			NameList: origNames,
			Type:     nextOrigType(),
			Values:   d.Values,
		}
		origSimd.SetPos(d.Pos())
		out = append(out, origSimd)
	}

	return out, true
}

func (r *Rewriter) rewriteSingleVarDecl(d *syntax.VarDecl, cloneOrigType bool) ([]syntax.Decl, bool) {
	name := d.NameList[0]
	obj := r.info.Defs[name]
	isDepType := obj != nil && r.analyzer.isDependentObj[obj]
	isDepInit := r.analyzer.hasNodeDependency(d.Values) || (d.Type != nil && r.analyzer.hasNodeDependency(d.Type))
	if !isDepType && !isDepInit {
		if cloneOrigType && d.Type != nil {
			d.Type = cloneExpr(d.Type)
		}
		return []syntax.Decl{d}, false
	}

	var out []syntax.Decl
	if isDepType {
		for _, k := range r.sizes {
			out = append(out, r.newCopier(k).CopyVarDecl(d))
		}
		if cloneOrigType && d.Type != nil {
			d.Type = cloneExpr(d.Type)
		}
		out = append(out, d)
		return out, true
	}

	// Otherwise, d's type does not depend on SIMD (or name is "_"), while its
	// initializer does: keep a single variable and emit @simdinit proxies.
	target := name.Value
	isBlank := target == "_"
	if isBlank {
		target = r.nextBlankTarget()
	}

	for _, k := range r.sizes {
		copier := r.newCopier(k)
		proxy := SimdInitProxyName(target, k)
		if base.Debug.Simd > 0 {
			base.Warn("%s: creating simd init proxy %s for %s", name.Pos().String(), proxy, name.Value)
		}
		specDecl := &syntax.VarDecl{
			Group:    d.Group,
			NameList: []*syntax.Name{syntax.NewName(name.Pos(), proxy)},
			Type:     copier.CopyExpr(d.Type),
			Values:   copier.CopyExpr(d.Values),
		}
		specDecl.SetPos(d.Pos())
		out = append(out, specDecl)
	}

	if !isBlank {
		lastSize := r.sizes[len(r.sizes)-1]
		lastProxyRef := syntax.NewName(name.Pos(), SimdInitProxyName(target, lastSize))
		origType := d.Type
		if cloneOrigType && origType != nil {
			origType = cloneExpr(origType)
		}
		origDecl := &syntax.VarDecl{
			Group:    d.Group,
			Pragma:   d.Pragma,
			NameList: []*syntax.Name{name},
			Type:     origType,
			Values:   lastProxyRef,
		}
		origDecl.SetPos(d.Pos())
		out = append(out, origDecl)
	}

	return out, true
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

func (r *Rewriter) isExcludedSimdFile(decl syntax.Decl) bool {
	// Files in package simd only declare stubs backed by simd/internal/bridge,
	// except for "tofrom_*" files, whose generic *FromArch functions live in
	// package simd itself and must be specialized by midway.
	if !r.analyzer.inSimd {
		return false
	}
	theFile := decl.Pos().Base().Filename()

	lastSlash := strings.LastIndex(theFile, SimdPkg+"/")
	lastBackslash := strings.LastIndex(theFile, SimdPkg+"\\")

	// Windows paths can be chaos, all we care, is whether the very last part
	// of the path is any-path-separator + "tofrom_" + anything-else, given that
	// we already know that we are in the simd package.
	maxSlash := max(lastSlash, lastBackslash)
	if maxSlash == -1 {
		return true
	}
	return !strings.HasPrefix(theFile[maxSlash:], SimdPkg+"/tofrom_") &&
		!strings.HasPrefix(theFile[maxSlash:], SimdPkg+"\\tofrom_")
}

func (r *Rewriter) shouldIncludeDecl(decl syntax.Decl) bool {
	if r.isExcludedSimdFile(decl) {
		return false
	}

	switch d := decl.(type) {
	case *syntax.FuncDecl:
		if d.Name != nil {
			return r.analyzer.isDependentObj[r.info.Defs[d.Name]]
		}
	case *syntax.TypeDecl:
		return r.analyzer.isDependentObj[r.info.Defs[d.Name]]
	}
	return false
}

// originals returns the names of the package-level types, variables, and
// functions that are originals, in sorted order. The methods of an original
// type are originals too, but aren't listed.
func (a *Analyzer) originals() []string {
	var names []string
	scope := a.pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if !a.isDependentObj[obj] {
			continue
		}
		if fn, ok := obj.(*types2.Func); ok && !a.HasDependentSignature(fn.Signature()) {
			// A boundary function. It keeps its name and
			// becomes the dispatcher.
			continue
		}
		names = append(names, name)
	}
	return names
}

// RewriteWrapper applies the midway rewrite to files, which are the type
// checked files of pkg.
//
// If any rewriting was performed, it returns a non-nil InitOrderPlan capturing
// the pre-rewrite Go-spec initialization order, which the caller must Apply to
// the re-checked types2.Info after typechecking finishes.
//
// It also returns the names of the package-level declarations that are
// originals (see the comment at the top of this file). Every method of an
// original type is also an original.
func RewriteWrapper(pkg *types2.Package, info *types2.Info, files []*syntax.File) (originals []string, initOrderPlan *InitOrderPlan) {
	if !buildcfg.Experiment.SIMDStable {
		return nil, nil
	}

	switch buildcfg.GOARCH {
	case "wasm", "amd64", "arm64":
	default:
		return nil, nil
	}

	sizes := rewriteSizes()
	if len(sizes) == 0 {
		return nil, nil
	}
	analyzer := NewAnalyzer(pkg, info)
	if !analyzer.Analyze(files) {
		return nil, nil
	}

	CheckPositions(files, "before midway")

	plan := saveInitOrder(info)
	rewriter := NewRewriter(pkg, info, analyzer, sizes)
	rewriter.Rewrite(files)

	CheckPositions(files, "after midway")

	return analyzer.originals(), plan
}
