// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package copylock defines an Analyzer that checks for locks
// erroneously passed by value.
package copylock

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/versions"
)

const Doc = `check for locks erroneously passed by value

Inadvertently copying a value containing a lock, such as sync.Mutex or
sync.WaitGroup, may cause both copies to malfunction. Generally such
values should be referred to through a pointer.`

var Analyzer = &analysis.Analyzer{
	Name:             "copylocks",
	Doc:              Doc,
	URL:              "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/copylock",
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	RunDespiteErrors: true,
	Run:              run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	var goversion string // effective file version ("" => unknown)
	nodeFilter := []ast.Node{
		(*ast.AssignStmt)(nil),
		(*ast.CallExpr)(nil),
		(*ast.CompositeLit)(nil),
		(*ast.File)(nil),
		(*ast.FuncDecl)(nil),
		(*ast.FuncLit)(nil),
		(*ast.GenDecl)(nil),
		(*ast.RangeStmt)(nil),
		(*ast.ReturnStmt)(nil),
	}
	inspect.WithStack(nodeFilter, func(node ast.Node, push bool, stack []ast.Node) bool {
		if !push {
			return false
		}
		switch node := node.(type) {
		case *ast.File:
			goversion = versions.FileVersion(pass.TypesInfo, node)
		case *ast.RangeStmt:
			checkCopyLocksRange(pass, node)
		case *ast.FuncDecl:
			checkCopyLocksFunc(pass, node.Name.Name, node.Recv, node.Type)
		case *ast.FuncLit:
			checkCopyLocksFunc(pass, "func", nil, node.Type)
		case *ast.CallExpr:
			checkCopyLocksCallExpr(pass, node)
		case *ast.AssignStmt:
			checkCopyLocksAssign(pass, node, goversion, parent(stack))
		case *ast.GenDecl:
			checkCopyLocksGenDecl(pass, node)
		case *ast.CompositeLit:
			checkCopyLocksCompositeLit(pass, node)
		case *ast.ReturnStmt:
			checkCopyLocksReturnStmt(pass, node)
		}
		return true
	})
	return nil, nil
}

// checkCopyLocksAssign checks whether an assignment
// copies a lock.
func checkCopyLocksAssign(pass *analysis.Pass, assign *ast.AssignStmt, goversion string, parent ast.Node) {
	lhs := assign.Lhs
	for i, x := range assign.Rhs {
		if path := lockPathRhs(pass, x); path != nil {
			pass.ReportRangef(x, "assignment copies lock value to %v: %v", analysisinternal.Format(pass.Fset, assign.Lhs[i]), path)
			lhs = nil // An lhs has been reported. We prefer the assignment warning and do not report twice.
		}
	}

	// After GoVersion 1.22, loop variables are implicitly copied on each iteration.
	// So a for statement may inadvertently copy a lock when any of the
	// iteration variables contain locks.
	if assign.Tok == token.DEFINE && versions.AtLeast(goversion, versions.Go1_22) {
		if parent, _ := parent.(*ast.ForStmt); parent != nil && parent.Init == assign {
			for _, l := range lhs {
				if id, ok := l.(*ast.Ident); ok && id.Name != "_" {
					if obj := pass.TypesInfo.Defs[id]; obj != nil && obj.Type() != nil {
						if path := lockPath(pass.Pkg, obj.Type(), nil); path != nil {
							pass.ReportRangef(l, "for loop iteration copies lock value to %v: %v", analysisinternal.Format(pass.Fset, l), path)
						}
					}
				}
			}
		}
	}
}

// checkCopyLocksGenDecl checks whether lock is copied
// in variable declaration.
func checkCopyLocksGenDecl(pass *analysis.Pass, gd *ast.GenDecl) {
	if gd.Tok != token.VAR {
		return
	}
	for _, spec := range gd.Specs {
		valueSpec := spec.(*ast.ValueSpec)
		for i, x := range valueSpec.Values {
			if path := lockPathRhs(pass, x); path != nil {
				pass.ReportRangef(x, "variable declaration copies lock value to %v: %v", valueSpec.Names[i].Name, path)
			}
		}
	}
}

// checkCopyLocksCompositeLit detects lock copy inside a composite literal
func checkCopyLocksCompositeLit(pass *analysis.Pass, cl *ast.CompositeLit) {
	for _, x := range cl.Elts {
		if node, ok := x.(*ast.KeyValueExpr); ok {
			x = node.Value
		}
		if path := lockPathRhs(pass, x); path != nil {
			pass.ReportRangef(x, "literal copies lock value from %v: %v", analysisinternal.Format(pass.Fset, x), path)
		}
	}
}

// checkCopyLocksReturnStmt detects lock copy in return statement
func checkCopyLocksReturnStmt(pass *analysis.Pass, rs *ast.ReturnStmt) {
	for _, x := range rs.Results {
		if path := lockPathRhs(pass, x); path != nil {
			pass.ReportRangef(x, "return copies lock value: %v", path)
		}
	}
}

// checkCopyLocksCallExpr detects lock copy in the arguments to a function call
func checkCopyLocksCallExpr(pass *analysis.Pass, ce *ast.CallExpr) {
	var id *ast.Ident
	switch fun := ce.Fun.(type) {
	case *ast.Ident:
		id = fun
	case *ast.SelectorExpr:
		id = fun.Sel
	}
	if fun, ok := pass.TypesInfo.Uses[id].(*types.Builtin); ok {
		switch fun.Name() {
		case "new", "len", "cap", "Sizeof", "Offsetof", "Alignof":
			return
		}
	}
	for _, x := range ce.Args {
		if path := lockPathRhs(pass, x); path != nil {
			pass.ReportRangef(x, "call of %s copies lock value: %v", analysisinternal.Format(pass.Fset, ce.Fun), path)
		}
	}
}

// checkCopyLocksFunc checks whether a function might
// inadvertently copy a lock, by checking whether
// its receiver, parameters, or return values
// are locks.
func checkCopyLocksFunc(pass *analysis.Pass, name string, recv *ast.FieldList, typ *ast.FuncType) {
	if recv != nil && len(recv.List) > 0 {
		expr := recv.List[0].Type
		if path := lockPath(pass.Pkg, pass.TypesInfo.Types[expr].Type, nil); path != nil {
			pass.ReportRangef(expr, "%s passes lock by value: %v", name, path)
		}
	}

	if typ.Params != nil {
		for _, field := range typ.Params.List {
			expr := field.Type
			if path := lockPath(pass.Pkg, pass.TypesInfo.Types[expr].Type, nil); path != nil {
				pass.ReportRangef(expr, "%s passes lock by value: %v", name, path)
			}
		}
	}

	// Don't check typ.Results. If T has a Lock field it's OK to write
	//     return T{}
	// because that is returning the zero value. Leave result checking
	// to the return statement.
}

// checkCopyLocksRange checks whether a range statement
// might inadvertently copy a lock by checking whether
// any of the range variables are locks.
func checkCopyLocksRange(pass *analysis.Pass, r *ast.RangeStmt) {
	checkCopyLocksRangeVar(pass, r.Tok, r.Key)
	checkCopyLocksRangeVar(pass, r.Tok, r.Value)
}

func checkCopyLocksRangeVar(pass *analysis.Pass, rtok token.Token, e ast.Expr) {
	if e == nil {
		return
	}
	id, isId := e.(*ast.Ident)
	if isId && id.Name == "_" {
		return
	}

	var typ types.Type
	if rtok == token.DEFINE {
		if !isId {
			return
		}
		obj := pass.TypesInfo.Defs[id]
		if obj == nil {
			return
		}
		typ = obj.Type()
	} else {
		typ = pass.TypesInfo.Types[e].Type
	}

	if typ == nil {
		return
	}
	if path := lockPath(pass.Pkg, typ, nil); path != nil {
		pass.Reportf(e.Pos(), "range var %s copies lock: %v", analysisinternal.Format(pass.Fset, e), path)
	}
}

type typePath []string

// String pretty-prints a typePath.
func (path typePath) String() string {
	n := len(path)
	var buf bytes.Buffer
	for i := range path {
		if i > 0 {
			fmt.Fprint(&buf, " contains ")
		}
		// The human-readable path is in reverse order, outermost to innermost.
		fmt.Fprint(&buf, path[n-i-1])
	}
	return buf.String()
}

func lockPathRhs(pass *analysis.Pass, x ast.Expr) typePath {
	x = ast.Unparen(x) // ignore parens on rhs

	if _, ok := x.(*ast.CompositeLit); ok {
		return nil
	}
	if _, ok := x.(*ast.CallExpr); ok {
		// A call may return a zero value.
		return nil
	}
	if star, ok := x.(*ast.StarExpr); ok {
		if _, ok := ast.Unparen(star.X).(*ast.CallExpr); ok {
			// A call may return a pointer to a zero value.
			return nil
		}
	}
	if tv, ok := pass.TypesInfo.Types[x]; ok && tv.IsValue() {
		return lockPath(pass.Pkg, tv.Type, nil)
	}
	return nil
}

// lockPath returns a typePath describing the location of a lock value
// contained in typ. If there is no contained lock, it returns nil.
//
// The seen map is used to short-circuit infinite recursion due to type cycles.
func lockPath(tpkg *types.Package, typ types.Type, seen map[types.Type]bool) typePath {
	if typ == nil || seen[typ] {
		return nil
	}
	if seen == nil {
		seen = make(map[types.Type]bool)
	}
	seen[typ] = true

	if tpar, ok := types.Unalias(typ).(*types.TypeParam); ok {
		terms, err := typeparams.StructuralTerms(tpar)
		if err != nil {
			return nil // invalid type
		}
		for _, term := range terms {
			subpath := lockPath(tpkg, term.Type(), seen)
			if len(subpath) > 0 {
				if term.Tilde() {
					// Prepend a tilde to our lock path entry to clarify the resulting
					// diagnostic message. Consider the following example:
					//
					//  func _[Mutex interface{ ~sync.Mutex; M() }](m Mutex) {}
					//
					// Here the naive error message will be something like "passes lock
					// by value: Mutex contains sync.Mutex". This is misleading because
					// the local type parameter doesn't actually contain sync.Mutex,
					// which lacks the M method.
					//
					// With tilde, it is clearer that the containment is via an
					// approximation element.
					subpath[len(subpath)-1] = "~" + subpath[len(subpath)-1]
				}
				return append(subpath, typ.String())
			}
		}
		return nil
	}

	for {
		atyp, ok := typ.Underlying().(*types.Array)
		if !ok {
			break
		}
		typ = atyp.Elem()
	}

	ttyp, ok := typ.Underlying().(*types.Tuple)
	if ok {
		for i := 0; i < ttyp.Len(); i++ {
			subpath := lockPath(tpkg, ttyp.At(i).Type(), seen)
			if subpath != nil {
				return append(subpath, typ.String())
			}
		}
		return nil
	}

	// We're only interested in the case in which the underlying
	// type is a struct. (Interfaces and pointers are safe to copy.)
	styp, ok := typ.Underlying().(*types.Struct)
	if !ok {
		return nil
	}

	// We're looking for cases in which a pointer to this type
	// is a sync.Locker, but a value is not. This differentiates
	// embedded interfaces from embedded values.
	if types.Implements(types.NewPointer(typ), lockerType) && !types.Implements(typ, lockerType) {
		return []string{typ.String()}
	}

	// In go1.10, sync.noCopy did not implement Locker.
	// (The Unlock method was added only in CL 121876.)
	// TODO(adonovan): remove workaround when we drop go1.10.
	if analysisinternal.IsTypeNamed(typ, "sync", "noCopy") {
		return []string{typ.String()}
	}

	nfields := styp.NumFields()
	for i := 0; i < nfields; i++ {
		ftyp := styp.Field(i).Type()
		subpath := lockPath(tpkg, ftyp, seen)
		if subpath != nil {
			return append(subpath, typ.String())
		}
	}

	return nil
}

// parent returns the second from the last node on stack if it exists.
func parent(stack []ast.Node) ast.Node {
	if len(stack) >= 2 {
		return stack[len(stack)-2]
	}
	return nil
}

var lockerType *types.Interface

// Construct a sync.Locker interface type.
func init() {
	nullary := types.NewSignature(nil, nil, nil, false) // func()
	methods := []*types.Func{
		types.NewFunc(token.NoPos, nil, "Lock", nullary),
		types.NewFunc(token.NoPos, nil, "Unlock", nullary),
	}
	lockerType = types.NewInterface(methods, nil).Complete()
}
