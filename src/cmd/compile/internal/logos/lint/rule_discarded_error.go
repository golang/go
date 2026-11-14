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

// discardedError is slop::discarded_error (SPEC Sec 1.2, #164): a
// statement that drops an error result: an expression statement, a call
// statement, an errdefer or scope defer statement, or a blank identifier
// that receives the result in an assignment or a variable declaration.
var discardedError = Rule{
	ID:    "slop::discarded_error",
	Group: Slop,
	Go:    true,
	Check: checkDiscardedError,
}

// discardCheck walks one file and reports dropped error results.
type discardCheck struct {
	pass   *Pass
	file   *syntax.File
	nilErr map[*types2.Func]bool
	// returnsError tells whether the function being walked returns error
	// as its last result.
	returnsError bool
	// group is the depth of go group blocks and child blocks around the
	// statement being walked; a go statement at depth above zero hands its
	// error to the group.
	group int
}

// checkDiscardedError is the Check of discardedError.
func checkDiscardedError(pass *Pass) {
	nilErr := facts.AlwaysNilErrorSet(pass.Pkg, pass.Info, pass.Files)
	for _, f := range pass.Files {
		c := &discardCheck{pass: pass, file: f, nilErr: nilErr}
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
		case *syntax.GroupStmt:
			c.walk(m.Ctx)
			if m.Limit != nil {
				c.walk(m.Limit)
			}
			c.groupBody(m.Body)
			return false
		case *syntax.GroupChild:
			c.groupBody(m.Body)
			return false
		case *syntax.ExprStmt:
			c.exprStmt(m)
		case *syntax.AssignStmt:
			c.assignStmt(m)
		case *syntax.VarDecl:
			c.varDecl(m)
		case *syntax.CallStmt:
			c.callStmt(m)
		case *syntax.ErrdeferStmt:
			c.dropCall(m.Call)
		case *syntax.ScopeDeferStmt:
			c.dropCall(m.Call)
		}
		return true
	})
}

func (c *discardCheck) walkFunc(body *syntax.BlockStmt, sig *syntax.FuncType) {
	outer := c.returnsError
	outerGroup := c.group
	c.returnsError = lastResultIsError(sig)
	c.group = 0
	c.walk(body)
	c.returnsError = outer
	c.group = outerGroup
}

func (c *discardCheck) exprStmt(s *syntax.ExprStmt) {
	call := callOf(s.X)
	if call == nil || !c.dropsError(call) {
		return
	}
	d := c.dropped(call)
	if c.returnsError && c.pass.Kind == KindLgo {
		if text, ok := c.sourceText(d.Pos, d.End); ok {
			d.Fixes = []Fix{{Description: "add ? to propagate the error", Replacement: text + "?"}}
		}
	}
	c.pass.Report(d)
}

// sourceText returns the source between start and end.
func (c *discardCheck) sourceText(start, end syntax.Pos) (string, bool) {
	src := c.pass.Source(c.file)
	from, okFrom := sourceOffset(src, start)
	to, okTo := sourceOffset(src, end)
	if !okFrom || !okTo || from > to {
		return "", false
	}
	return string(src[from:to]), true
}

func (c *discardCheck) groupBody(body *syntax.BlockStmt) {
	c.group++
	c.walk(body)
	c.group--
}

func (c *discardCheck) callStmt(s *syntax.CallStmt) {
	if s.Tok == syntax.Go && c.group > 0 {
		return
	}
	c.dropCall(s.Call)
}

func (c *discardCheck) dropCall(e syntax.Expr) {
	call := callOf(e)
	if call == nil || !c.dropsError(call) {
		return
	}
	c.pass.Report(c.dropped(call))
}

func (c *discardCheck) dropsError(call *syntax.CallExpr) bool {
	if c.benignCall(call) {
		return false
	}
	return slices.ContainsFunc(elemTypes(TypeOf(call)), isErrorType)
}

// assignStmt reports a blank identifier that receives an error result.
func (c *discardCheck) assignStmt(s *syntax.AssignStmt) {
	if s.Rhs == nil || (s.Op != 0 && s.Op != syntax.Def) {
		return
	}
	c.blankReceive(exprList(s.Lhs), exprList(s.Rhs))
}

// varDecl reports a blank identifier that receives an error result.
func (c *discardCheck) varDecl(d *syntax.VarDecl) {
	if lhs, rhs, ok := declPairs(d); ok {
		c.blankReceive(lhs, rhs)
	}
}

func (c *discardCheck) blankReceive(lhs, rhs []syntax.Expr) {
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
		if call == nil || !isBlank(lhs[i]) || c.benignCall(call) {
			continue
		}
		if res := elemTypes(TypeOf(call)); len(res) == 1 && isErrorType(res[0]) {
			return call
		}
	}
	return nil
}

// tupleBlankCall returns the call whose tuple result has an error value
// at the position of a blank identifier.
func (c *discardCheck) tupleBlankCall(lhs []syntax.Expr, e syntax.Expr) *syntax.CallExpr {
	call := callOf(e)
	if call == nil || c.benignCall(call) {
		return nil
	}
	res := elemTypes(TypeOf(call))
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

func (c *discardCheck) benignCall(call *syntax.CallExpr) bool {
	fn := calleeFunc(c.pass, call)
	if fn == nil {
		return false
	}
	switch name := fn.FullName(); name {
	case "fmt.Print", "fmt.Printf", "fmt.Println":
		return true
	case "fmt.Fprint", "fmt.Fprintf", "fmt.Fprintln":
		return len(call.ArgList) > 0 && (isStdoutOrStderr(c.pass, call.ArgList[0]) ||
			isBufferOrBuilder(c.pass, call.ArgList[0]))
	case "io.WriteString":
		return len(call.ArgList) > 0 && isBufferOrBuilder(c.pass, call.ArgList[0])
	default:
		if strings.HasPrefix(name, "(*bytes.Buffer).Write") ||
			strings.HasPrefix(name, "(*strings.Builder).Write") || isHashWrite(c.pass, call) {
			return true
		}
	}
	return c.alwaysNil(fn)
}

func (c *discardCheck) alwaysNil(fn *types2.Func) bool {
	if c.nilErr[fn] {
		return true
	}
	if fn.Pkg() == nil || fn.Pkg() == c.pass.Pkg {
		return false
	}
	return facts.AlwaysNilCallee(fn)
}

// isHashWrite reports whether call writes on a receiver that implements
// hash.Hash. Such a Write never returns an error.
func isHashWrite(pass *Pass, call *syntax.CallExpr) bool {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Write" {
		return false
	}
	hash := hashInterface(pass.Pkg)
	if hash == nil {
		return false
	}
	recv := TypeOf(sel.X)
	if recv == nil {
		return false
	}
	if types2.Implements(recv, hash) {
		return true
	}
	_, isPtr := types2.Unalias(recv).Underlying().(*types2.Pointer)
	return !isPtr && types2.Implements(types2.NewPointer(recv), hash)
}

// hashInterface returns the hash.Hash interface visible from pkg over its
// transitive imports, or nil when no import chain reaches it.
func hashInterface(pkg *types2.Package) *types2.Interface {
	if pkg == nil {
		return nil
	}
	hash := findPackage(pkg, "hash")
	if hash == nil {
		return nil
	}
	obj, ok := hash.Scope().Lookup("Hash").(*types2.TypeName)
	if !ok {
		return nil
	}
	iface, ok := obj.Type().Underlying().(*types2.Interface)
	if !ok {
		return nil
	}
	return iface
}

// findPackage returns the transitively imported package with the given path,
// or nil.
func findPackage(root *types2.Package, path string) *types2.Package {
	seen := map[*types2.Package]bool{root: true}
	queue := []*types2.Package{root}
	for len(queue) > 0 {
		pkg := queue[0]
		queue = queue[1:]
		if pkg.Path() == path {
			return pkg
		}
		for _, imp := range pkg.Imports() {
			if !seen[imp] {
				seen[imp] = true
				queue = append(queue, imp)
			}
		}
	}
	return nil
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

func isBufferOrBuilder(pass *Pass, w syntax.Expr) bool {
	ptr, ok := types2.Unalias(TypeOf(w)).(*types2.Pointer)
	if !ok {
		return false
	}
	named, ok := types2.Unalias(ptr.Elem()).(*types2.Named)
	if !ok {
		return false
	}
	obj := named.Obj()
	if obj == nil || obj.Pkg() == nil {
		return false
	}
	return (obj.Pkg().Path() == "bytes" && obj.Name() == "Buffer") ||
		(obj.Pkg().Path() == "strings" && obj.Name() == "Builder")
}
