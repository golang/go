// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
	"slices"
)

type LogosConstFor struct {
	Target    Type
	Struct    *Struct
	FieldVar  *Var
	IndexVar  *Var
	Generic   bool
	GoVersion string
}

type LogosCopyContext struct {
	Scope      *Scope
	Sig        *Signature
	Version    string
	Disqual    []map[string]bool
	Funcs      []*Func
	Aliases    map[*Package]string
	MaxSteps   int
	NodeLimit  int
	Importer   Importer
	Context    *Context
	ConstFuncs map[string]LogosConstFunc
	Bind       LogosTypeBindings
}

func (ctx LogosCopyContext) logosCopyChecker(pkg *Package, info *Info, errs *[]Error) *Checker {
	imp := ctx.Importer
	if imp == nil && info != nil {
		imp = info.LogosImporter
	}
	ctxt := ctx.Context
	if ctxt == nil && info != nil {
		ctxt = info.LogosContext
	}
	if ctxt == nil {
		ctxt = NewContext()
	}
	conf := &Config{
		Logos:              true,
		GoVersion:          ctx.Version,
		Context:            ctxt,
		Importer:           imp,
		IgnoreBranchErrors: true,
		LogosCTFEMaxSteps:  ctx.MaxSteps,
		Error: func(err error) {
			if e, ok := err.(Error); ok && !e.Soft && errs != nil {
				*errs = append(*errs, e)
			}
		},
	}
	conf.LogosComptimeNodeLimit = ctx.NodeLimit
	if conf.LogosCTFEMaxSteps == 0 && info != nil {
		conf.LogosCTFEMaxSteps = info.LogosCTFEMaxSteps
	}
	mc := NewChecker(conf, pkg, info)
	mc.version = asGoVersion(ctx.Version)
	mc.logosCopy = true
	mc.scope = NewScope(ctx.Scope, nopos, nopos, "logos copies")
	for _, f := range ctx.Funcs {
		mc.scope.Insert(f)
	}
	for imported, alias := range ctx.Aliases {
		mc.scope.Insert(NewPkgName(nopos, pkg, alias, imported))
	}
	mc.sig = ctx.Sig
	mc.logosDisqual = slices.Clone(ctx.Disqual)
	mc.logosSeedConstFuncs(info, ctx.ConstFuncs)
	return mc
}

func (check *Checker) logosSeedConstFuncs(info *Info, explicit map[string]LogosConstFunc) {
	if len(explicit) == 0 && (info == nil || len(info.LogosConstFuncs) == 0) {
		return
	}
	bodies, consts := check.logosConstFuncBodies()
	seed := func(name string, rec LogosConstFunc) {
		if rec.Decl == nil {
			return
		}
		if _, dup := bodies[name]; !dup {
			bodies[name] = rec.Decl
		}
		if rec.Const {
			consts[rec.Decl] = true
		}
	}
	if info != nil {
		for name, rec := range info.LogosConstFuncs {
			seed(name, rec)
		}
	}
	for name, rec := range explicit {
		if rec.Decl == nil {
			continue
		}
		bodies[name] = rec.Decl
		if rec.Const {
			consts[rec.Decl] = true
		}
	}
}

func (check *Checker) logosCopyFinish() {
	check.processDelayed(0)
	check.cleanup()
	check.recordUntyped()
}

func LogosCheckCopy(pkg *Package, info *Info, ctx LogosCopyContext, blocks []*syntax.BlockStmt) {
	if ctx.Scope == nil || ctx.Sig == nil {
		return
	}
	mc := ctx.logosCopyChecker(pkg, info, nil)
	for _, b := range blocks {
		mc.stmt(0, b)
	}
	mc.logosCopyFinish()
}

func LogosCheckCopyFunc(pkg *Package, info *Info, ctx LogosCopyContext, fn *syntax.FuncDecl) (*Func, []Error) {
	if ctx.Scope == nil || fn.Body == nil {
		return nil, nil
	}
	var errs []Error
	mc := ctx.logosCopyChecker(pkg, info, &errs)
	sig := new(Signature)
	mc.funcType(sig, nil, nil, fn.Type)
	obj := NewFunc(fn.Name.Pos(), pkg, fn.Name.Value, sig)
	mc.recordDef(fn.Name, obj)
	mc.scope.Insert(obj)
	mc.funcBody(nil, fn.Name.Value, sig, fn.Body, nil)
	mc.logosCopyFinish()
	return obj, errs
}

type LogosTypeBindings struct {
	Types  map[string]Type
	Pkgs   map[string]*Package
	Fields map[string]Type
}

func LogosResolveType(pkg *Package, info *Info, ctx LogosCopyContext, e syntax.Expr) (Type, bool) {
	if ctx.Scope == nil {
		return nil, false
	}
	var errs []Error
	mc := ctx.logosCopyChecker(pkg, info, &errs)
	for name, t := range ctx.Bind.Types {
		mc.scope.Insert(NewTypeName(nopos, pkg, name, t))
	}
	for name, p := range ctx.Bind.Pkgs {
		mc.scope.Insert(NewPkgName(nopos, pkg, name, p))
	}
	for name, t := range ctx.Bind.Fields {
		stub := NewPackage("", name)
		stub.Scope().Insert(NewTypeName(nopos, stub, "Type", t))
		mc.scope.Insert(NewPkgName(nopos, pkg, name, stub))
	}
	t := mc.typ(e)
	mc.logosCopyFinish()
	return t, len(errs) == 0 && isValid(t)
}

type logosFieldBinding struct {
	info  *LogosConstFor
	index int
	field *Var
	outer *logosFieldBinding
	tag   string
}

var logosFieldValueSels = []string{"Name", "Index", "Tag", "Exported"}

func (check *Checker) logosConstForIndexValue(x *operand, obj Object) {
	if check.logosPred.depth > 0 {
		return
	}
	if n := len(check.logosLHS); n > 0 && check.logosLHS[n-1] == obj {
		return
	}
	for b := check.logosFieldBind; b != nil; b = b.outer {
		if b.info.IndexVar != nil && obj == b.info.IndexVar && x.isValid() {
			x.mode_ = constant_
			x.val = constant.MakeInt64(int64(b.index))
			return
		}
	}
}

func (check *Checker) logosConstFor(s *syntax.ConstForStmt, ctxt stmtContext) {
	info := &LogosConstFor{GoVersion: check.conf.GoVersion}
	if check.logosConstForDepth == 0 {
		check.logosConstForBudget = 0
	}
	check.logosConstForDepth++
	defer func() { check.logosConstForDepth-- }()
	if check.logosConstForRange(s, info) {
		check.logosConstForBody(s, info, ctxt)
	}
	if check.Info.LogosConstFor == nil {
		check.Info.LogosConstFor = make(map[syntax.Node]*LogosConstFor)
	}
	check.Info.LogosConstFor[s] = info
}

func (check *Checker) logosConstForRange(s *syntax.ConstForStmt, info *LogosConstFor) bool {
	bad := func(pos syntax.Pos, format string, args ...any) bool {
		check.errorf(pos, InvalidRangeExpr, "[comptime::const_for_fields] "+format, args...)
		return false
	}
	call, ok := s.Range.(*syntax.CallExpr)
	if !ok {
		return bad(s.Range.Pos(), "const for range must be comptime.Fields[T]()")
	}
	if len(call.ArgList) != 0 {
		return bad(s.Range.Pos(), "const for range must be comptime.Fields[T]()")
	}
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok {
		return bad(s.Range.Pos(), "const for range must be comptime.Fields[T]()")
	}
	targs := syntax.UnpackListExpr(idx.Index)
	if len(targs) != 1 {
		return bad(s.Range.Pos(), "const for range must be comptime.Fields[T]()")
	}
	targExpr := targs[0]
	sel, fname, ok := check.logosComptimeQualifier(call)
	if !ok || fname != "Fields" {
		return bad(s.Range.Pos(), "const for range must be comptime.Fields[T]()")
	}
	check.recordUse(sel.Sel, check.logosComptimeObj("Fields"))
	if name, ok := sel.X.(*syntax.Name); ok {
		if _, obj := check.lookupScope(name.Value); obj != nil {
			check.recordUse(name, obj)
			if pn, ok := obj.(*PkgName); ok {
				check.usedPkgNames[pn] = true
			}
		}
	}
	targ := check.varType(targExpr)
	if !isValid(targ) {
		return false
	}
	if tp, ok := targ.(*TypeParam); ok {
		_ = tp
		info.Target = targ
		info.Generic = true
		return true
	}
	st, ok := targ.Underlying().(*Struct)
	if !ok && check.Info.LogosConstIfDead[s] {
		return false
	}
	if !ok {
		return bad(targExpr.Pos(), "const for range must be a struct type, have %s", targ)
	}
	info.Target = targ
	info.Struct = st
	return true
}

func (check *Checker) logosComptimeObj(name string) Object {
	return logosComptimePackage().Scope().Lookup(name)
}

func (check *Checker) logosConstForBody(s *syntax.ConstForStmt, info *LogosConstFor, ctxt stmtContext) {
	check.openScope(s, "const for")
	defer check.closeScope()

	fieldTyp := logosComptimeAny
	if obj := check.logosComptimeObj("Field"); obj != nil {
		fieldTyp = obj.Type()
	}
	fv := NewVar(s.Field.Pos(), check.pkg, s.Field.Value, fieldTyp)
	check.declare(check.scope, s.Field, fv, s.Field.Pos())
	info.FieldVar = fv
	if check.logosFieldVars == nil {
		check.logosFieldVars = make(map[*Var]*LogosConstFor)
	}
	check.logosFieldVars[fv] = info
	if s.Index != nil && s.Index.Value != "_" {
		iv := NewVar(s.Index.Pos(), check.pkg, s.Index.Value, Typ[Int])
		check.declare(check.scope, s.Index, iv, s.Index.Pos())
		info.IndexVar = iv
	}

	inner := ctxt &^ (fallthroughOk | finalSwitchCase | inTypeSwitch)
	if info.Struct != nil {
		check.logosConstForConcrete(s, info, inner)
	} else if info.Generic {
		check.stmtList(inner, s.Body.List)
	}
	check.logosFieldDrain(s, info)
}

func (check *Checker) logosConstForConcrete(s *syntax.ConstForStmt, info *LogosConstFor, inner stmtContext) {
	saved := check.conf.Error
	seen := make(map[string]bool)
	outer := check.logosFieldBind
	defer func() { check.logosFieldBind = outer }()
	check.conf.Error = func(err error) {
		if key := err.Error(); !seen[key] {
			seen[key] = true
			saved(err)
		}
	}
	defer func() { check.conf.Error = saved }()
	checked := 0
	bodyNodes := logosCountBodyNodes(s.Body)
	for i := range info.Struct.NumFields() {
		if !check.logosConstForCharge(s, bodyNodes) {
			break
		}
		f := info.Struct.Field(i)
		if !f.Exported() && f.Pkg() != check.pkg {
			continue
		}
		checked++
		check.logosFieldBind = &logosFieldBinding{info: info, index: i, field: f, tag: info.Struct.Tag(i), outer: outer}
		check.openScope(s.Body, "const for iteration")
		check.stmtList(inner, s.Body.List)
		check.closeScope()
	}
	if checked == 0 {
		check.usedVars[info.FieldVar] = true
		if info.IndexVar != nil {
			check.usedVars[info.IndexVar] = true
		}
	}
}

func (check *Checker) logosConstForCharge(s *syntax.ConstForStmt, n int) bool {
	if check.logosConstForBudget > 8*check.logosNodeLimit() {
		return false
	}
	check.logosConstForBudget += n
	if check.logosConstForBudget > 8*check.logosNodeLimit() {
		check.errorf(s.Pos(), InvalidRangeExpr, "[comptime::node_limit] const for has %d iterations above the selected limit %d", check.logosConstForBudget, check.logosNodeLimit())
		return false
	}
	return true
}

func (check *Checker) logosNodeLimit() int {
	if check.conf.LogosComptimeNodeLimit > 0 {
		return check.conf.LogosComptimeNodeLimit
	}
	return 4096
}

func logosCountBodyNodes(root syntax.Node) int {
	n := 0
	syntax.Inspect(root, func(x syntax.Node) bool {
		n++
		return true
	})
	if n < 1 {
		return 1
	}
	return n
}

func (check *Checker) logosFieldDrain(s *syntax.ConstForStmt, info *LogosConstFor) {
	exempt := make(map[*syntax.Name]bool)
	syntax.Inspect(s.Body, func(n syntax.Node) bool {
		as, ok := n.(*syntax.AssignStmt)
		if !ok || as.Op != 0 {
			return true
		}
		lhs := syntax.UnpackListExpr(as.Lhs)
		rhs := syntax.UnpackListExpr(as.Rhs)
		if len(lhs) != 1 || len(rhs) != 1 {
			return true
		}
		lname, ok := lhs[0].(*syntax.Name)
		if !ok || lname.Value != "_" {
			return true
		}
		if rhs, ok := rhs[0].(*syntax.Name); ok {
			exempt[rhs] = true
		}
		return true
	})
	for name, v := range check.logosFieldPend {
		if v != info.FieldVar || exempt[name] {
			continue
		}
		check.errorf(name.Pos(), InvalidPkgUse, "[comptime::runtime_use] field variable %s is compile-time only", name.Value)
		check.usedVars[v] = true
		delete(check.logosFieldPend, name)
	}
}

func (check *Checker) logosFieldVarNote(e *syntax.Name, obj Object) {
	v, ok := obj.(*Var)
	if !ok || check.logosFieldVars[v] == nil {
		return
	}
	if check.logosFieldPend == nil {
		check.logosFieldPend = make(map[*syntax.Name]*Var)
	}
	check.logosFieldPend[e] = v
}

func (check *Checker) logosFieldVarSelMark(x *operand, e *syntax.SelectorExpr) {
	name, ok := e.X.(*syntax.Name)
	if !ok {
		return
	}
	_, obj := check.lookupScope(name.Value)
	v, ok := obj.(*Var)
	if !ok || check.logosFieldVars[v] == nil {
		return
	}
	for _, sel := range logosFieldValueSels {
		if e.Sel.Value == sel {
			delete(check.logosFieldPend, name)
			return
		}
	}
	if e.Sel.Value != "Type" {
		delete(check.logosFieldPend, name)
	}
}

func (check *Checker) logosFieldTypeType(e *syntax.SelectorExpr, x *operand) bool {
	name, ok := e.X.(*syntax.Name)
	if !ok || e.Sel.Value != "Type" {
		return false
	}
	_, obj := check.lookupScope(name.Value)
	v, ok := obj.(*Var)
	if !ok || check.logosFieldVars[v] == nil {
		return false
	}
	delete(check.logosFieldPend, name)
	check.recordUse(name, obj)
	check.usedVars[v] = true
	x.expr = e
	if b := check.logosFieldBind; b != nil && check.logosFieldVars[v] == b.info {
		x.mode_ = typexpr
		x.typ_ = b.field.Type()
		return true
	}
	x.mode_ = typexpr
	x.typ_ = logosComptimeAny
	return true
}

func (check *Checker) logosFieldProjExpr(x *operand, e *syntax.FieldProjExpr) {
	x.invalidate()
	x.expr = e
	_, obj := check.lookupScope(e.Field.Value)
	v, ok := obj.(*Var)
	info := check.logosFieldVars[v]
	if !ok || info == nil {
		check.errorf(e.Field.Pos(), MissingFieldOrMethod, "[comptime::const_for_projection] %s is not a const for field variable", e.Field.Value)
		return
	}
	delete(check.logosFieldPend, e.Field)
	check.recordUse(e.Field, v)
	check.usedVars[v] = true
	var xop operand
	check.expr(nil, &xop, e.X)
	if !xop.isValid() {
		return
	}
	xt := Unalias(xop.typ())
	mode := operandMode(value)
	if xop.mode() == variable {
		mode = variable
	}
	if pt, ok := xt.(*Pointer); ok {
		xt = Unalias(pt.Elem())
		mode = variable
	}
	if _, ok := xt.(*TypeParam); ok {
		if info.Generic {
			x.mode_ = mode
			x.typ_ = logosComptimeAny
			return
		}
		check.errorf(e.Pos(), MissingFieldOrMethod, "[comptime::const_for_projection] cannot project a field of %s", xop.typ())
		return
	}
	st, ok := xt.Underlying().(*Struct)
	if !ok || info.Struct == nil || st != info.Struct {
		check.errorf(e.Pos(), MissingFieldOrMethod, "[comptime::const_for_projection] cannot project a field of %s", xop.typ())
		return
	}
	x.mode_ = mode
	if b := check.logosFieldBind; b != nil && info == b.info {
		x.typ_ = b.field.Type()
		return
	}
	x.typ_ = check.logosFieldProjType(info)
}

func (check *Checker) logosFieldProjType(info *LogosConstFor) Type {
	var shared Type
	for i := range info.Struct.NumFields() {
		t := info.Struct.Field(i).Type()
		if shared == nil {
			shared = t
		} else if !Identical(shared, t) {
			return logosComptimeAny
		}
	}
	if shared == nil {
		return logosComptimeAny
	}
	return shared
}
