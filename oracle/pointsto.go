// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/oracle/serial"
)

// pointsto runs the pointer analysis on the selected expression,
// and reports its points-to set (for a pointer-like expression)
// or its dynamic types (for an interface, reflect.Value, or
// reflect.Type expression) and their points-to sets.
//
// All printed sets are sorted to ensure determinism.
//
func pointsto(q *Query) error {
	lconf := loader.Config{Build: q.Build}

	if err := setPTAScope(&lconf, q.Scope); err != nil {
		return err
	}

	// Load/parse/type-check the program.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}
	q.Fset = lprog.Fset

	qpos, err := parseQueryPos(lprog, q.Pos, true) // needs exact pos
	if err != nil {
		return err
	}

	prog := ssautil.CreateProgram(lprog, ssa.GlobalDebug)

	ptaConfig, err := setupPTA(prog, lprog, q.PTALog, q.Reflection)
	if err != nil {
		return err
	}

	path, action := findInterestingNode(qpos.info, qpos.path)
	if action != actionExpr {
		return fmt.Errorf("pointer analysis wants an expression; got %s",
			astutil.NodeDescription(qpos.path[0]))
	}

	var expr ast.Expr
	var obj types.Object
	switch n := path[0].(type) {
	case *ast.ValueSpec:
		// ambiguous ValueSpec containing multiple names
		return fmt.Errorf("multiple value specification")
	case *ast.Ident:
		obj = qpos.info.ObjectOf(n)
		expr = n
	case ast.Expr:
		expr = n
	default:
		// TODO(adonovan): is this reachable?
		return fmt.Errorf("unexpected AST for expr: %T", n)
	}

	// Reject non-pointerlike types (includes all constants---except nil).
	// TODO(adonovan): reject nil too.
	typ := qpos.info.TypeOf(expr)
	if !pointer.CanPoint(typ) {
		return fmt.Errorf("pointer analysis wants an expression of reference type; got %s", typ)
	}

	// Determine the ssa.Value for the expression.
	var value ssa.Value
	var isAddr bool
	if obj != nil {
		// def/ref of func/var object
		value, isAddr, err = ssaValueForIdent(prog, qpos.info, obj, path)
	} else {
		value, isAddr, err = ssaValueForExpr(prog, qpos.info, path)
	}
	if err != nil {
		return err // e.g. trivially dead code
	}

	// Defer SSA construction till after errors are reported.
	prog.Build()

	// Run the pointer analysis.
	ptrs, err := runPTA(ptaConfig, value, isAddr)
	if err != nil {
		return err // e.g. analytically unreachable
	}

	q.result = &pointstoResult{
		qpos: qpos,
		typ:  typ,
		ptrs: ptrs,
	}
	return nil
}

// ssaValueForIdent returns the ssa.Value for the ast.Ident whose path
// to the root of the AST is path.  isAddr reports whether the
// ssa.Value is the address denoted by the ast.Ident, not its value.
//
func ssaValueForIdent(prog *ssa.Program, qinfo *loader.PackageInfo, obj types.Object, path []ast.Node) (value ssa.Value, isAddr bool, err error) {
	switch obj := obj.(type) {
	case *types.Var:
		pkg := prog.Package(qinfo.Pkg)
		pkg.Build()
		if v, addr := prog.VarValue(obj, pkg, path); v != nil {
			return v, addr, nil
		}
		return nil, false, fmt.Errorf("can't locate SSA Value for var %s", obj.Name())

	case *types.Func:
		fn := prog.FuncValue(obj)
		if fn == nil {
			return nil, false, fmt.Errorf("%s is an interface method", obj)
		}
		// TODO(adonovan): there's no point running PTA on a *Func ident.
		// Eliminate this feature.
		return fn, false, nil
	}
	panic(obj)
}

// ssaValueForExpr returns the ssa.Value of the non-ast.Ident
// expression whose path to the root of the AST is path.
//
func ssaValueForExpr(prog *ssa.Program, qinfo *loader.PackageInfo, path []ast.Node) (value ssa.Value, isAddr bool, err error) {
	pkg := prog.Package(qinfo.Pkg)
	pkg.SetDebugMode(true)
	pkg.Build()

	fn := ssa.EnclosingFunction(pkg, path)
	if fn == nil {
		return nil, false, fmt.Errorf("no SSA function built for this location (dead code?)")
	}

	if v, addr := fn.ValueForExpr(path[0].(ast.Expr)); v != nil {
		return v, addr, nil
	}

	return nil, false, fmt.Errorf("can't locate SSA Value for expression in %s", fn)
}

// runPTA runs the pointer analysis of the selected SSA value or address.
func runPTA(conf *pointer.Config, v ssa.Value, isAddr bool) (ptrs []pointerResult, err error) {
	T := v.Type()
	if isAddr {
		conf.AddIndirectQuery(v)
		T = deref(T)
	} else {
		conf.AddQuery(v)
	}
	ptares := ptrAnalysis(conf)

	var ptr pointer.Pointer
	if isAddr {
		ptr = ptares.IndirectQueries[v]
	} else {
		ptr = ptares.Queries[v]
	}
	if ptr == (pointer.Pointer{}) {
		return nil, fmt.Errorf("pointer analysis did not find expression (dead code?)")
	}
	pts := ptr.PointsTo()

	if pointer.CanHaveDynamicTypes(T) {
		// Show concrete types for interface/reflect.Value expression.
		if concs := pts.DynamicTypes(); concs.Len() > 0 {
			concs.Iterate(func(conc types.Type, pta interface{}) {
				labels := pta.(pointer.PointsToSet).Labels()
				sort.Sort(byPosAndString(labels)) // to ensure determinism
				ptrs = append(ptrs, pointerResult{conc, labels})
			})
		}
	} else {
		// Show labels for other expressions.
		labels := pts.Labels()
		sort.Sort(byPosAndString(labels)) // to ensure determinism
		ptrs = append(ptrs, pointerResult{T, labels})
	}
	sort.Sort(byTypeString(ptrs)) // to ensure determinism
	return ptrs, nil
}

type pointerResult struct {
	typ    types.Type       // type of the pointer (always concrete)
	labels []*pointer.Label // set of labels
}

type pointstoResult struct {
	qpos *queryPos
	typ  types.Type      // type of expression
	ptrs []pointerResult // pointer info (typ is concrete => len==1)
}

func (r *pointstoResult) display(printf printfFunc) {
	if pointer.CanHaveDynamicTypes(r.typ) {
		// Show concrete types for interface, reflect.Type or
		// reflect.Value expression.

		if len(r.ptrs) > 0 {
			printf(r.qpos, "this %s may contain these dynamic types:", r.qpos.typeString(r.typ))
			for _, ptr := range r.ptrs {
				var obj types.Object
				if nt, ok := deref(ptr.typ).(*types.Named); ok {
					obj = nt.Obj()
				}
				if len(ptr.labels) > 0 {
					printf(obj, "\t%s, may point to:", r.qpos.typeString(ptr.typ))
					printLabels(printf, ptr.labels, "\t\t")
				} else {
					printf(obj, "\t%s", r.qpos.typeString(ptr.typ))
				}
			}
		} else {
			printf(r.qpos, "this %s cannot contain any dynamic types.", r.typ)
		}
	} else {
		// Show labels for other expressions.
		if ptr := r.ptrs[0]; len(ptr.labels) > 0 {
			printf(r.qpos, "this %s may point to these objects:",
				r.qpos.typeString(r.typ))
			printLabels(printf, ptr.labels, "\t")
		} else {
			printf(r.qpos, "this %s may not point to anything.",
				r.qpos.typeString(r.typ))
		}
	}
}

func (r *pointstoResult) toSerial(res *serial.Result, fset *token.FileSet) {
	var pts []serial.PointsTo
	for _, ptr := range r.ptrs {
		var namePos string
		if nt, ok := deref(ptr.typ).(*types.Named); ok {
			namePos = fset.Position(nt.Obj().Pos()).String()
		}
		var labels []serial.PointsToLabel
		for _, l := range ptr.labels {
			labels = append(labels, serial.PointsToLabel{
				Pos:  fset.Position(l.Pos()).String(),
				Desc: l.String(),
			})
		}
		pts = append(pts, serial.PointsTo{
			Type:    r.qpos.typeString(ptr.typ),
			NamePos: namePos,
			Labels:  labels,
		})
	}
	res.PointsTo = pts
}

type byTypeString []pointerResult

func (a byTypeString) Len() int           { return len(a) }
func (a byTypeString) Less(i, j int) bool { return a[i].typ.String() < a[j].typ.String() }
func (a byTypeString) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

type byPosAndString []*pointer.Label

func (a byPosAndString) Len() int { return len(a) }
func (a byPosAndString) Less(i, j int) bool {
	cmp := a[i].Pos() - a[j].Pos()
	return cmp < 0 || (cmp == 0 && a[i].String() < a[j].String())
}
func (a byPosAndString) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

func printLabels(printf printfFunc, labels []*pointer.Label, prefix string) {
	// TODO(adonovan): due to context-sensitivity, many of these
	// labels may differ only by context, which isn't apparent.
	for _, label := range labels {
		printf(label, "%s%s", prefix, label)
	}
}
