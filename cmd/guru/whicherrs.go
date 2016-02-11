// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"

	"golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

var builtinErrorType = types.Universe.Lookup("error").Type()

// whicherrs takes an position to an error and tries to find all types, constants
// and global value which a given error can point to and which can be checked from the
// scope where the error lives.
// In short, it returns a list of things that can be checked against in order to handle
// an error properly.
//
// TODO(dmorsing): figure out if fields in errors like *os.PathError.Err
// can be queried recursively somehow.
func whicherrs(q *Query) error {
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
		return fmt.Errorf("whicherrs wants an expression; got %s",
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
		return fmt.Errorf("unexpected AST for expr: %T", n)
	}

	typ := qpos.info.TypeOf(expr)
	if !types.Identical(typ, builtinErrorType) {
		return fmt.Errorf("selection is not an expression of type 'error'")
	}
	// Determine the ssa.Value for the expression.
	var value ssa.Value
	if obj != nil {
		// def/ref of func/var object
		value, _, err = ssaValueForIdent(prog, qpos.info, obj, path)
	} else {
		value, _, err = ssaValueForExpr(prog, qpos.info, path)
	}
	if err != nil {
		return err // e.g. trivially dead code
	}

	// Defer SSA construction till after errors are reported.
	prog.Build()

	globals := findVisibleErrs(prog, qpos)
	constants := findVisibleConsts(prog, qpos)

	res := &whicherrsResult{
		qpos:   qpos,
		errpos: expr.Pos(),
	}

	// TODO(adonovan): the following code is heavily duplicated
	// w.r.t.  "pointsto".  Refactor?

	// Find the instruction which initialized the
	// global error. If more than one instruction has stored to the global
	// remove the global from the set of values that we want to query.
	allFuncs := ssautil.AllFunctions(prog)
	for fn := range allFuncs {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				store, ok := instr.(*ssa.Store)
				if !ok {
					continue
				}
				gval, ok := store.Addr.(*ssa.Global)
				if !ok {
					continue
				}
				gbl, ok := globals[gval]
				if !ok {
					continue
				}
				// we already found a store to this global
				// The normal error define is just one store in the init
				// so we just remove this global from the set we want to query
				if gbl != nil {
					delete(globals, gval)
				}
				globals[gval] = store.Val
			}
		}
	}

	ptaConfig.AddQuery(value)
	for _, v := range globals {
		ptaConfig.AddQuery(v)
	}

	ptares := ptrAnalysis(ptaConfig)
	valueptr := ptares.Queries[value]
	for g, v := range globals {
		ptr, ok := ptares.Queries[v]
		if !ok {
			continue
		}
		if !ptr.MayAlias(valueptr) {
			continue
		}
		res.globals = append(res.globals, g)
	}
	pts := valueptr.PointsTo()
	dedup := make(map[*ssa.NamedConst]bool)
	for _, label := range pts.Labels() {
		// These values are either MakeInterfaces or reflect
		// generated interfaces. For the purposes of this
		// analysis, we don't care about reflect generated ones
		makeiface, ok := label.Value().(*ssa.MakeInterface)
		if !ok {
			continue
		}
		constval, ok := makeiface.X.(*ssa.Const)
		if !ok {
			continue
		}
		c := constants[*constval]
		if c != nil && !dedup[c] {
			dedup[c] = true
			res.consts = append(res.consts, c)
		}
	}
	concs := pts.DynamicTypes()
	concs.Iterate(func(conc types.Type, _ interface{}) {
		// go/types is a bit annoying here.
		// We want to find all the types that we can
		// typeswitch or assert to. This means finding out
		// if the type pointed to can be seen by us.
		//
		// For the purposes of this analysis, the type is always
		// either a Named type or a pointer to one.
		// There are cases where error can be implemented
		// by unnamed types, but in that case, we can't assert to
		// it, so we don't care about it for this analysis.
		var name *types.TypeName
		switch t := conc.(type) {
		case *types.Pointer:
			named, ok := t.Elem().(*types.Named)
			if !ok {
				return
			}
			name = named.Obj()
		case *types.Named:
			name = t.Obj()
		default:
			return
		}
		if !isAccessibleFrom(name, qpos.info.Pkg) {
			return
		}
		res.types = append(res.types, &errorType{conc, name})
	})
	sort.Sort(membersByPosAndString(res.globals))
	sort.Sort(membersByPosAndString(res.consts))
	sort.Sort(sorterrorType(res.types))

	q.result = res
	return nil
}

// findVisibleErrs returns a mapping from each package-level variable of type "error" to nil.
func findVisibleErrs(prog *ssa.Program, qpos *queryPos) map[*ssa.Global]ssa.Value {
	globals := make(map[*ssa.Global]ssa.Value)
	for _, pkg := range prog.AllPackages() {
		for _, mem := range pkg.Members {
			gbl, ok := mem.(*ssa.Global)
			if !ok {
				continue
			}
			gbltype := gbl.Type()
			// globals are always pointers
			if !types.Identical(deref(gbltype), builtinErrorType) {
				continue
			}
			if !isAccessibleFrom(gbl.Object(), qpos.info.Pkg) {
				continue
			}
			globals[gbl] = nil
		}
	}
	return globals
}

// findVisibleConsts returns a mapping from each package-level constant assignable to type "error", to nil.
func findVisibleConsts(prog *ssa.Program, qpos *queryPos) map[ssa.Const]*ssa.NamedConst {
	constants := make(map[ssa.Const]*ssa.NamedConst)
	for _, pkg := range prog.AllPackages() {
		for _, mem := range pkg.Members {
			obj, ok := mem.(*ssa.NamedConst)
			if !ok {
				continue
			}
			consttype := obj.Type()
			if !types.AssignableTo(consttype, builtinErrorType) {
				continue
			}
			if !isAccessibleFrom(obj.Object(), qpos.info.Pkg) {
				continue
			}
			constants[*obj.Value] = obj
		}
	}

	return constants
}

type membersByPosAndString []ssa.Member

func (a membersByPosAndString) Len() int { return len(a) }
func (a membersByPosAndString) Less(i, j int) bool {
	cmp := a[i].Pos() - a[j].Pos()
	return cmp < 0 || cmp == 0 && a[i].String() < a[j].String()
}
func (a membersByPosAndString) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

type sorterrorType []*errorType

func (a sorterrorType) Len() int { return len(a) }
func (a sorterrorType) Less(i, j int) bool {
	cmp := a[i].obj.Pos() - a[j].obj.Pos()
	return cmp < 0 || cmp == 0 && a[i].typ.String() < a[j].typ.String()
}
func (a sorterrorType) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

type errorType struct {
	typ types.Type      // concrete type N or *N that implements error
	obj *types.TypeName // the named type N
}

type whicherrsResult struct {
	qpos    *queryPos
	errpos  token.Pos
	globals []ssa.Member
	consts  []ssa.Member
	types   []*errorType
}

func (r *whicherrsResult) display(printf printfFunc) {
	if len(r.globals) > 0 {
		printf(r.qpos, "this error may point to these globals:")
		for _, g := range r.globals {
			printf(g.Pos(), "\t%s", g.RelString(r.qpos.info.Pkg))
		}
	}
	if len(r.consts) > 0 {
		printf(r.qpos, "this error may contain these constants:")
		for _, c := range r.consts {
			printf(c.Pos(), "\t%s", c.RelString(r.qpos.info.Pkg))
		}
	}
	if len(r.types) > 0 {
		printf(r.qpos, "this error may contain these dynamic types:")
		for _, t := range r.types {
			printf(t.obj.Pos(), "\t%s", r.qpos.typeString(t.typ))
		}
	}
}

func (r *whicherrsResult) toSerial(res *serial.Result, fset *token.FileSet) {
	we := &serial.WhichErrs{}
	we.ErrPos = fset.Position(r.errpos).String()
	for _, g := range r.globals {
		we.Globals = append(we.Globals, fset.Position(g.Pos()).String())
	}
	for _, c := range r.consts {
		we.Constants = append(we.Constants, fset.Position(c.Pos()).String())
	}
	for _, t := range r.types {
		var et serial.WhichErrsType
		et.Type = r.qpos.typeString(t.typ)
		et.Position = fset.Position(t.obj.Pos()).String()
		we.Types = append(we.Types, et)
	}
	res.WhichErrs = we
}
