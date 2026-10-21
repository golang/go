// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"slices"
)

// LoGo/s M7-S2 `const for` checking (SPEC Sec 2.5, T-205). stmt.go calls
// in through one hook line; lowering lives in logos/constfor.go.
//
// A `const for` unrolls once per field of its struct type in declaration
// order. Over a concrete struct the body is checked once per field with
// the field binding installed, so every unrolled body is fully verified
// (#76): f.Name is a constant, f.Type is the exact field type, and
// val.[f] carries the exact field type. Over a type parameter (M7-S3)
// the body is checked once with the field type approximated as any; the
// per-instantiation copies are verified by the recheck pass (#128).

// LogosConstFor is the recorded meaning of one checked ConstForStmt.
type LogosConstFor struct {
	Target    Type
	Struct    *Struct
	FieldVar  *Var
	IndexVar  *Var
	Generic   bool
	GoVersion string
}

// LogosCopyContext carries the enclosing function state one copy check
// runs under: the scope that holds the copied code, the enclosing
// signature (statement copies only), the language version, the
// narrowing-disqualified sets the main check recorded for the enclosing
// body, so copies narrow exactly like the source body, the
// comptime-generic copies emitted so far, which later copies may call,
// and the file's synthesized package aliases (the `_logos_tN` qualifiers
// that spell a type of a package the file does not import), each bound to
// the package object it names.
type LogosCopyContext struct {
	Scope   *Scope
	Sig     *Signature
	Version string
	Disqual []map[string]bool
	Funcs   []*Func
	Aliases map[*Package]string
}

// logosCopyChecker builds the throwaway checker one copy check runs on.
// It appends every hard diagnostic to *errs, or drops it when errs is
// nil. Soft diagnostics (unused variables and labels, warn-only rules)
// never block a fact, and the recheck of the lowered tree reports them.
// The parser already checked the copy's branches and labels.
func (ctx LogosCopyContext) logosCopyChecker(pkg *Package, info *Info, errs *[]Error) *Checker {
	conf := &Config{
		Logos:              true,
		GoVersion:          ctx.Version,
		Context:            NewContext(),
		IgnoreBranchErrors: true,
		Error: func(err error) {
			if e, ok := err.(Error); ok && !e.Soft && errs != nil {
				*errs = append(*errs, e)
			}
		},
	}
	mc := NewChecker(conf, pkg, info)
	mc.version = asGoVersion(ctx.Version)
	mc.scope = NewScope(ctx.Scope, nopos, nopos, "logos copies")
	for _, f := range ctx.Funcs {
		mc.scope.Insert(f)
	}
	for imported, alias := range ctx.Aliases {
		mc.scope.Insert(NewPkgName(nopos, pkg, alias, imported))
	}
	mc.sig = ctx.Sig
	mc.logosDisqual = slices.Clone(ctx.Disqual)
	return mc
}

// logosCopyFinish runs the checker's delayed work and records the
// untyped expressions of one copy check.
func (check *Checker) logosCopyFinish() {
	check.processDelayed(0)
	check.cleanup()
	check.recordUntyped()
}

// LogosCheckCopy checks copied statements under ctx, recording types,
// uses, definitions, selections, instances, and scopes into info for
// the lowering of one const for unrolling. It reports no diagnostic: the
// statements copy the body of a concrete const for, whose body the main
// check already verified once per field (#76), so a copy cannot hold a
// new defect. The checker here lacks the enclosing function's flow state
// (active nil narrowings, unchecked error results), so a diagnostic it
// raised would be provisional, and the recheck of the lowered tree
// reports the defects that remain.
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

// LogosCheckCopyFunc checks one copied function declaration (an
// instantiation of a comptime generic) under ctx, recording its types
// into info, and returns the function object so later copies can call
// it, with the hard diagnostics of the check. The definition checked
// with its type parameters approximated, so the copy is the first check
// of the instantiated body, and it checks as a whole function: no state
// of an enclosing function is missing, so every hard diagnostic is
// real. It returns nil when the declaration has no body or ctx holds no
// scope.
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

// logosFieldBinding is the per-iteration field meaning installed while
// checking one unrolled body over a concrete struct.
type logosFieldBinding struct {
	info  *LogosConstFor
	index int
	field *Var
	tag   string
}

var logosFieldValueSels = []string{"Name", "Index", "Tag", "Exported"}

// logosConstFor checks a const for statement.
func (check *Checker) logosConstFor(s *syntax.ConstForStmt, ctxt stmtContext) {
	info := &LogosConstFor{GoVersion: check.conf.GoVersion}
	if check.logosConstForRange(s, info) {
		check.logosConstForBody(s, info, ctxt)
	}
	if check.Info.LogosConstFor == nil {
		check.Info.LogosConstFor = make(map[syntax.Node]*LogosConstFor)
	}
	check.Info.LogosConstFor[s] = info
}

// logosConstForRange resolves the range operand. It reports false after
// an error, with info left unusable.
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
	sel, fname, ok := check.logosComptimeRangeFun(call)
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

// logosComptimeRangeFun unwraps a range call's selector when it names a
// virtual comptime function.
func (check *Checker) logosComptimeRangeFun(call *syntax.CallExpr) (*syntax.SelectorExpr, string, bool) {
	fun := call.Fun
	if idx, ok := fun.(*syntax.IndexExpr); ok {
		fun = idx.X
	}
	sel, ok := fun.(*syntax.SelectorExpr)
	if !ok {
		return nil, "", false
	}
	name, ok := sel.X.(*syntax.Name)
	if !ok {
		return nil, "", false
	}
	_, obj := check.lookupScope(name.Value)
	pn, ok := obj.(*PkgName)
	if !ok || pn.Imported() != logosComptimePkg {
		return nil, "", false
	}
	return sel, sel.Sel.Value, true
}

// logosComptimeObj returns the synthetic object with the given name.
func (check *Checker) logosComptimeObj(name string) Object {
	return logosComptimePackage().Scope().Lookup(name)
}

// logosConstForBody declares the loop variables and checks the body.
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

// logosConstForConcrete checks the body once per field with the binding
// installed (#76: every unrolled body passes the full checks). One
// defect that every iteration reproduces is reported once: an identical
// diagnostic of a later iteration is dropped, a diagnostic that differs
// per field is reported.
func (check *Checker) logosConstForConcrete(s *syntax.ConstForStmt, info *LogosConstFor, inner stmtContext) {
	saved := check.conf.Error
	seen := make(map[string]bool)
	check.conf.Error = func(err error) {
		if key := err.Error(); !seen[key] {
			seen[key] = true
			saved(err)
		}
	}
	defer func() { check.conf.Error = saved }()
	checked := 0
	for i := range info.Struct.NumFields() {
		f := info.Struct.Field(i)
		if !f.Exported() && f.Pkg() != check.pkg {
			continue
		}
		checked++
		check.logosFieldBind = &logosFieldBinding{info: info, index: i, field: f, tag: info.Struct.Tag(i)}
		check.openScope(s.Body, "const for iteration")
		check.stmtList(inner, s.Body.List)
		check.closeScope()
	}
	check.logosFieldBind = nil
	if checked == 0 {
		check.usedVars[info.FieldVar] = true
		if info.IndexVar != nil {
			check.usedVars[info.IndexVar] = true
		}
	}
}

// logosFieldDrain reports bare field-variable uses left pending after
// the body check. A `_ = f` blank assignment carries no value and is
// exempt.
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

// logosFieldVarNote records a bare use of a field variable for the
// drain. Supported shapes remove their nodes when they check.
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

// logosFieldVarSelMark consumes supported value selections on a field
// variable. It reports whether Sel is a compile-time field name.
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

// logosFieldTypeType substitutes f.Type in type position: the exact
// field type with a binding installed, any over a bare type parameter.
// It reports false for any other selector.
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

// logosFieldProjExpr checks a val.[f] projection.
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

// logosFieldProjType is the first-check result type of a projection over
// a concrete struct: the shared field type, or any when fields differ.
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
