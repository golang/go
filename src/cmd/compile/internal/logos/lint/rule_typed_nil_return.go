// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// typedNilReturn is correctness::typed_nil_return (SPEC Sec 1.2, T-102): a
// return statement hands a possibly nil pointer to an interface or error
// result without a dominating nil check. An interface that holds a nil
// pointer is not nil, so the caller's err != nil passes on it. Three
// operands count:
//
//   - the conversion (*T)(nil), which is always nil and so never checked;
//   - an option value that the checker has not narrowed, because a none
//     boxes into a non-nil interface;
//   - a platform pointer, which the checker records in
//     Info.LogosPlatformRets: a pointer from a Go declaration, whose nil
//     contract the callee keeps.
//
// A nil check dominates when an if or for condition on the same path
// expression holds around the return, or an earlier if on a nil test of it
// exits the block (see nilFacts). A call result is never checked, so a
// returned call that yields an option or a platform pointer is reported.
// Only the (*T)(nil) form is active in a Go package, because options and
// platform pointers do not exist there. The rule is forbid in a loGo/s
// package and deny in a Go package.
var typedNilReturn = Rule{
	ID:    "correctness::typed_nil_return",
	Group: Correctness,
	Go:    true,
	Check: checkTypedNilReturn,
}

// typedNil is the analysis of one function.
type typedNil struct {
	pass    *Pass
	file    *syntax.File
	results *types2.Tuple
}

// checkTypedNilReturn is the Check of typedNilReturn.
func checkTypedNilReturn(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncDecl:
				if fn, ok := pass.Info.Defs[n.Name].(*types2.Func); ok && n.Body != nil {
					analyzeReturns(pass, f, fn.Signature(), n.Body)
				}
			case *syntax.FuncLit:
				if sig, ok := TypeOf(n).(*types2.Signature); ok {
					analyzeReturns(pass, f, sig, n.Body)
				}
			}
			return true
		})
	}
}

// analyzeReturns checks the return statements of one function body. A
// function with no interface result has nothing to box.
func analyzeReturns(pass *Pass, f *syntax.File, sig *types2.Signature, body *syntax.BlockStmt) {
	t := &typedNil{pass: pass, file: f, results: sig.Results()}
	if !t.hasBoxingResult() {
		return
	}
	w := &flow{pass: pass, onReturn: t.returnStmt}
	w.stmts(body.List, nilFacts{})
}

// hasBoxingResult reports whether some result of the function is an
// interface.
func (t *typedNil) hasBoxingResult() bool {
	for i := range t.results.Len() {
		if boxes(t.results.At(i).Type()) {
			return true
		}
	}
	return false
}

// boxes reports whether a value stored into a result of type slot is boxed
// into an interface: an interface type that is not a type parameter.
func boxes(slot types2.Type) bool {
	if _, ok := types2.Unalias(slot).(*types2.TypeParam); ok {
		return false
	}
	return types2.IsInterface(slot)
}

// returnStmt checks the results of one return statement against the result
// slots, with the facts that hold before it. A single call that supplies all
// the results is checked once, against every slot it fills.
func (t *typedNil) returnStmt(s *syntax.ReturnStmt, facts nilFacts) {
	if s.Results == nil {
		return
	}
	list := exprList(s.Results)
	if len(list) == t.results.Len() {
		for i, e := range list {
			t.operand(e, t.results.At(i).Type(), facts)
		}
		return
	}
	if call := callOf(list[0]); len(list) == 1 && call != nil {
		t.tuple(call, facts)
	}
}

// tuple checks a call returned in place of the whole result list.
func (t *typedNil) tuple(call *syntax.CallExpr, facts nilFacts) {
	types := resultTypes(call)
	for i := range min(len(types), t.results.Len()) {
		if boxes(t.results.At(i).Type()) && t.reportOperand(call, types[i], t.results.At(i).Type(), facts) {
			return
		}
	}
}

// operand checks one result expression stored into a slot.
func (t *typedNil) operand(e syntax.Expr, slot types2.Type, facts nilFacts) {
	if boxes(slot) {
		t.reportOperand(e, TypeOf(e), slot, facts)
	}
}

// reportOperand reports e, of type typ, stored into the interface slot when
// it is a possibly nil pointer with no dominating check. It reports whether
// it did.
func (t *typedNil) reportOperand(e syntax.Expr, typ, slot types2.Type, facts nilFacts) bool {
	qual := types2.RelativeTo(t.pass.Pkg)
	result := types2.TypeString(slot, qual)
	if conv := nilConversion(t.pass, e); conv != nil {
		t.report(conv, "%s returned as %s result is a typed nil: the interface is not nil; return nil instead", syntax.String(conv), result)
		return true
	}
	if t.pass.Kind != KindLgo || facts[pathKey(e)] {
		return false
	}
	switch {
	case optionOf(typ) != nil:
		t.report(e, "option %s (%s) returned as %s result without a dominating nil check: a none value boxes into a non-nil interface", syntax.String(e), types2.TypeString(typ, qual), result)
	case t.platformFlow(e, typ, slot):
		t.report(e, "platform pointer %s (%s) returned as %s result without a dominating nil check: a nil pointer boxes into a non-nil interface", syntax.String(e), types2.TypeString(typ, qual), result)
	default:
		return false
	}
	return true
}

// platformFlow reports whether the checker recorded the operand e, of type
// typ, as a platform pointer converted to the interface result type slot.
func (t *typedNil) platformFlow(e syntax.Expr, typ, slot types2.Type) bool {
	for _, r := range t.pass.Info.LogosPlatformRets[syntax.StartPos(e)] {
		if types2.Identical(r.Value, typ) && types2.Identical(r.Slot, slot) {
			return true
		}
	}
	return false
}

// report records a finding over e.
func (t *typedNil) report(e syntax.Expr, format string, args ...any) {
	d := Diagnostic{Pos: syntax.StartPos(e), End: syntax.EndPos(e), Message: fmt.Sprintf(format, args...)}
	if call, ok := e.(*syntax.CallExpr); ok {
		d.End = t.pass.CallEnd(t.file, call)
	}
	t.pass.Report(d)
}

// nilConversion returns the conversion (*T)(nil) that e is, looking through
// parentheses, or nil.
func nilConversion(pass *Pass, e syntax.Expr) *syntax.CallExpr {
	call := callOf(e)
	if call == nil || len(call.ArgList) != 1 || !call.Fun.GetTypeInfo().IsType() {
		return nil
	}
	typ := TypeOf(call)
	if typ == nil || !isNilLiteral(pass, call.ArgList[0]) {
		return nil
	}
	if _, ok := typ.Underlying().(*types2.Pointer); !ok {
		return nil
	}
	return call
}

// isNilLiteral reports whether e is the predeclared nil.
func isNilLiteral(pass *Pass, e syntax.Expr) bool {
	n, ok := syntax.Unparen(e).(*syntax.Name)
	if !ok {
		return false
	}
	_, isNil := pass.Info.Uses[n].(*types2.Nil)
	return isNil
}

// optionOf returns the option type t denotes, looking through aliases and
// defined types, or nil.
func optionOf(t types2.Type) *types2.Option {
	if t == nil {
		return nil
	}
	o, _ := types2.Unalias(t).Underlying().(*types2.Option)
	return o
}
