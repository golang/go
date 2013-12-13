// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"

	"code.google.com/p/go.tools/astutil"
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/oracle/serial"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// pointsto runs the pointer analysis on the selected expression,
// and reports its points-to set (for a pointer-like expression)
// or its dynamic types (for an interface, reflect.Value, or
// reflect.Type expression) and their points-to sets.
//
// All printed sets are sorted to ensure determinism.
//
func pointsto(o *Oracle, qpos *QueryPos) (queryResult, error) {
	path, action := findInterestingNode(qpos.info, qpos.path)
	if action != actionExpr {
		return nil, fmt.Errorf("pointer analysis wants an expression; got %s",
			astutil.NodeDescription(qpos.path[0]))
	}

	var expr ast.Expr
	var obj types.Object
	switch n := path[0].(type) {
	case *ast.ValueSpec:
		// ambiguous ValueSpec containing multiple names
		return nil, fmt.Errorf("multiple value specification")
	case *ast.Ident:
		obj = qpos.info.ObjectOf(n)
		expr = n
	case ast.Expr:
		expr = n
	default:
		// TODO(adonovan): is this reachable?
		return nil, fmt.Errorf("unexpected AST for expr: %T", n)
	}

	// Reject non-pointerlike types (includes all constants).
	typ := qpos.info.TypeOf(expr)
	if !pointer.CanPoint(typ) {
		return nil, fmt.Errorf("pointer analysis wants an expression of reference type; got %s", typ)
	}

	// Determine the ssa.Value for the expression.
	var value ssa.Value
	var isAddr bool
	var err error
	if obj != nil {
		// def/ref of func/var object
		value, isAddr, err = ssaValueForIdent(o.prog, qpos.info, obj, path)
	} else {
		value, isAddr, err = ssaValueForExpr(o.prog, qpos.info, path)
	}
	if err != nil {
		return nil, err // e.g. trivially dead code
	}

	// Run the pointer analysis.
	ptrs, err := runPTA(o, value, isAddr)
	if err != nil {
		return nil, err // e.g. analytically unreachable
	}

	return &pointstoResult{
		qpos: qpos,
		typ:  typ,
		ptrs: ptrs,
	}, nil
}

// ssaValueForIdent returns the ssa.Value for the ast.Ident whose path
// to the root of the AST is path.  isAddr reports whether the
// ssa.Value is the address denoted by the ast.Ident, not its value.
//
func ssaValueForIdent(prog *ssa.Program, qinfo *importer.PackageInfo, obj types.Object, path []ast.Node) (value ssa.Value, isAddr bool, err error) {
	switch obj := obj.(type) {
	case *types.Var:
		pkg := prog.Package(qinfo.Pkg)
		pkg.Build()
		if v, addr := prog.VarValue(obj, pkg, path); v != nil {
			return v, addr, nil
		}
		return nil, false, fmt.Errorf("can't locate SSA Value for var %s", obj.Name())

	case *types.Func:
		return prog.FuncValue(obj), false, nil
	}
	panic(obj)
}

// ssaValueForExpr returns the ssa.Value of the non-ast.Ident
// expression whose path to the root of the AST is path.
//
func ssaValueForExpr(prog *ssa.Program, qinfo *importer.PackageInfo, path []ast.Node) (value ssa.Value, isAddr bool, err error) {
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
func runPTA(o *Oracle, v ssa.Value, isAddr bool) (ptrs []pointerResult, err error) {
	buildSSA(o)

	if isAddr {
		o.ptaConfig.AddIndirectQuery(v)
	} else {
		o.ptaConfig.AddQuery(v)
	}
	ptares := ptrAnalysis(o)

	// Combine the PT sets from all contexts.
	var pointers []pointer.Pointer
	if isAddr {
		pointers = ptares.IndirectQueries[v]
	} else {
		pointers = ptares.Queries[v]
	}
	if pointers == nil {
		return nil, fmt.Errorf("pointer analysis did not find expression (dead code?)")
	}
	pts := pointer.PointsToCombined(pointers)

	if pointer.CanHaveDynamicTypes(v.Type()) {
		// Show concrete types for interface/reflect.Value expression.
		if concs := pts.DynamicTypes(); concs.Len() > 0 {
			concs.Iterate(func(conc types.Type, pta interface{}) {
				combined := pointer.PointsToCombined(pta.([]pointer.Pointer))
				labels := combined.Labels()
				sort.Sort(byPosAndString(labels)) // to ensure determinism
				ptrs = append(ptrs, pointerResult{conc, labels})
			})
		}
	} else {
		// Show labels for other expressions.
		labels := pts.Labels()
		sort.Sort(byPosAndString(labels)) // to ensure determinism
		ptrs = append(ptrs, pointerResult{v.Type(), labels})
	}
	sort.Sort(byTypeString(ptrs)) // to ensure determinism
	return ptrs, nil
}

type pointerResult struct {
	typ    types.Type       // type of the pointer (always concrete)
	labels []*pointer.Label // set of labels
}

type pointstoResult struct {
	qpos *QueryPos
	typ  types.Type      // type of expression
	ptrs []pointerResult // pointer info (typ is concrete => len==1)
}

func (r *pointstoResult) display(printf printfFunc) {
	if pointer.CanHaveDynamicTypes(r.typ) {
		// Show concrete types for interface, reflect.Type or
		// reflect.Value expression.

		if len(r.ptrs) > 0 {
			printf(r.qpos, "this %s may contain these dynamic types:", r.qpos.TypeString(r.typ))
			for _, ptr := range r.ptrs {
				var obj types.Object
				if nt, ok := deref(ptr.typ).(*types.Named); ok {
					obj = nt.Obj()
				}
				if len(ptr.labels) > 0 {
					printf(obj, "\t%s, may point to:", r.qpos.TypeString(ptr.typ))
					printLabels(printf, ptr.labels, "\t\t")
				} else {
					printf(obj, "\t%s", r.qpos.TypeString(ptr.typ))
				}
			}
		} else {
			printf(r.qpos, "this %s cannot contain any dynamic types.", r.typ)
		}
	} else {
		// Show labels for other expressions.
		if ptr := r.ptrs[0]; len(ptr.labels) > 0 {
			printf(r.qpos, "this %s may point to these objects:",
				r.qpos.TypeString(r.typ))
			printLabels(printf, ptr.labels, "\t")
		} else {
			printf(r.qpos, "this %s may not point to anything.",
				r.qpos.TypeString(r.typ))
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
			Type:    r.qpos.TypeString(ptr.typ),
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
