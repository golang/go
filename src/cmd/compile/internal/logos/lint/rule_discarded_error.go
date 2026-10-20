// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// discardedError is slop::discarded_error (SPEC Sec 1.2, #164): an
// expression statement or a blank assignment that drops an error result.
// Calls that cannot fail in practice are exempt: callees whose package facts
// carry AlwaysNilError, fmt.Print*, fmt.Fprint* to os.Stdout or os.Stderr,
// and the Write* methods of *bytes.Buffer and *strings.Builder. No producer
// of the AlwaysNilError fact exists yet, so that exemption applies only to
// facts the importer has recorded.
var discardedError = Rule{
	ID:    "slop::discarded_error",
	Group: Slop,
	Go:    true,
	Check: checkDiscardedError,
}

// discardCheck walks one file and reports dropped error results.
type discardCheck struct {
	pass *Pass
	file *syntax.File
	// returnsError tells whether the function being walked returns error
	// as its last result.
	returnsError bool
}

// checkDiscardedError is the Check of discardedError.
func checkDiscardedError(pass *Pass) {
	for _, f := range pass.Files {
		c := &discardCheck{pass: pass, file: f}
		c.walk(f)
	}
}

// walk visits the statements under n. A function starts a new walk with its
// own signature.
func (c *discardCheck) walk(n syntax.Node) {
	syntax.Inspect(n, func(m syntax.Node) bool {
		switch m := m.(type) {
		case *syntax.FuncDecl:
			if m.Body != nil {
				c.walkFunc(m.Body, m.Type)
			}
			return false
		case *syntax.FuncLit:
			c.walkFunc(m.Body, m.Type)
			return false
		case *syntax.ExprStmt:
			c.exprStmt(m)
		case *syntax.AssignStmt:
			c.assignStmt(m)
		}
		return true
	})
}

// walkFunc walks the body of a function with the signature sig.
func (c *discardCheck) walkFunc(body *syntax.BlockStmt, sig *syntax.FuncType) {
	outer := c.returnsError
	c.returnsError = lastResultIsError(sig)
	c.walk(body)
	c.returnsError = outer
}

// exprStmt reports a call statement that drops an error result.
func (c *discardCheck) exprStmt(s *syntax.ExprStmt) {
	call := callOf(s.X)
	if call == nil || benignCall(c.pass, call) {
		return
	}
	if !slices.ContainsFunc(resultTypes(call), isErrorType) {
		return
	}
	d := c.dropped(call)
	if c.returnsError && c.pass.Kind == KindLgo {
		d.Fixes = []Fix{{Description: "add ? to propagate the error", Replacement: syntax.String(call) + "?"}}
	}
	c.pass.Report(d)
}

// assignStmt reports a blank identifier that receives an error result.
func (c *discardCheck) assignStmt(s *syntax.AssignStmt) {
	if s.Rhs == nil || (s.Op != 0 && s.Op != syntax.Def) {
		return
	}
	lhs, rhs := exprList(s.Lhs), exprList(s.Rhs)
	var call *syntax.CallExpr
	switch {
	case len(lhs) == len(rhs):
		call = c.pairedBlankCall(lhs, rhs)
	case len(rhs) == 1:
		call = c.tupleBlankCall(lhs, rhs[0])
	}
	if call != nil {
		c.pass.Report(c.dropped(call))
	}
}

// pairedBlankCall returns the call whose single error result a blank
// identifier receives in a one-to-one assignment.
func (c *discardCheck) pairedBlankCall(lhs, rhs []syntax.Expr) *syntax.CallExpr {
	for i := range lhs {
		call := callOf(rhs[i])
		if call == nil || !isBlank(lhs[i]) || benignCall(c.pass, call) {
			continue
		}
		if res := resultTypes(call); len(res) == 1 && isErrorType(res[0]) {
			return call
		}
	}
	return nil
}

// tupleBlankCall returns the call whose tuple result has an error value
// at the position of a blank identifier.
func (c *discardCheck) tupleBlankCall(lhs []syntax.Expr, e syntax.Expr) *syntax.CallExpr {
	call := callOf(e)
	if call == nil || benignCall(c.pass, call) {
		return nil
	}
	res := resultTypes(call)
	if len(res) != len(lhs) {
		return nil
	}
	for i, l := range lhs {
		if isBlank(l) && isErrorType(res[i]) {
			return call
		}
	}
	return nil
}

// dropped builds the finding for a call whose error is dropped. Its span is
// the call, through the closing parenthesis.
func (c *discardCheck) dropped(call *syntax.CallExpr) Diagnostic {
	text := syntax.String(call)
	return Diagnostic{
		Pos:     syntax.StartPos(call),
		End:     c.pass.CallEnd(c.file, call),
		Message: "error result of " + text + " is dropped; handle or propagate it",
		Provenance: Provenance{
			OriginExpr:   text,
			OriginLine:   int(call.Pos().Line()),
			InferredType: "error",
		},
	}
}

// lastResultIsError reports whether the last result of sig is error.
func lastResultIsError(sig *syntax.FuncType) bool {
	if sig == nil || len(sig.ResultList) == 0 {
		return false
	}
	return isErrorType(TypeOf(sig.ResultList[len(sig.ResultList)-1].Type))
}

// benignCall reports whether call is one of the exempt callees.
func benignCall(pass *Pass, call *syntax.CallExpr) bool {
	fn := calleeFunc(pass, call)
	if fn == nil {
		return false
	}
	switch name := fn.FullName(); name {
	case "fmt.Print", "fmt.Printf", "fmt.Println":
		return true
	case "fmt.Fprint", "fmt.Fprintf", "fmt.Fprintln":
		return len(call.ArgList) > 0 && isStdoutOrStderr(pass, call.ArgList[0])
	default:
		return strings.HasPrefix(name, "(*bytes.Buffer).Write") ||
			strings.HasPrefix(name, "(*strings.Builder).Write") ||
			alwaysNilError(fn)
	}
}

// alwaysNilError reports whether the facts recorded for the package of fn
// mark it AlwaysNilError.
func alwaysNilError(fn *types2.Func) bool {
	if fn.Pkg() == nil {
		return false
	}
	f, ok := facts.Lookup(fn.Pkg().Path())
	if !ok || f == nil {
		return false
	}
	object := factObject(fn)
	return slices.ContainsFunc(f.Fns, func(ff facts.FnFact) bool {
		return ff.Object == object && ff.AlwaysNilError
	})
}

// factObject returns the name of fn in a facts file: Name, Type.Method, or
// (*Type).Method.
func factObject(fn *types2.Func) string {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return fn.Name()
	}
	recv := types2.Unalias(sig.Recv().Type())
	ptr, isPtr := recv.(*types2.Pointer)
	if isPtr {
		recv = types2.Unalias(ptr.Elem())
	}
	named, ok := recv.(*types2.Named)
	if !ok {
		return fn.Name()
	}
	if isPtr {
		return "(*" + named.Obj().Name() + ")." + fn.Name()
	}
	return named.Obj().Name() + "." + fn.Name()
}

// isStdoutOrStderr reports whether w is os.Stdout or os.Stderr.
func isStdoutOrStderr(pass *Pass, w syntax.Expr) bool {
	sel, ok := w.(*syntax.SelectorExpr)
	if !ok {
		return false
	}
	if v, ok := pass.Info.Uses[sel.Sel].(*types2.Var); ok && v.Pkg() != nil && v.Pkg().Path() == "os" {
		return v.Name() == "Stdout" || v.Name() == "Stderr"
	}
	return false
}
