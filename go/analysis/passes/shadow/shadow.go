// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package shadow

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

// NOTE: Experimental. Not part of the vet suite.

const Doc = `check for possible unintended shadowing of variables

This analyzer check for shadowed variables.
A shadowed variable is a variable declared in an inner scope
with the same name and type as a variable in an outer scope,
and where the outer variable is mentioned after the inner one
is declared.

(This definition can be refined; the module generates too many
false positives and is not yet enabled by default.)

For example:

	func BadRead(f *os.File, buf []byte) error {
		var err error
		for {
			n, err := f.Read(buf) // shadows the function variable 'err'
			if err != nil {
				break // causes return of wrong value
			}
			foo(buf)
		}
		return err
	}
`

var Analyzer = &analysis.Analyzer{
	Name:     "shadow",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// flags
var strict = false

func init() {
	Analyzer.Flags.BoolVar(&strict, "strict", strict, "whether to be strict about shadowing; can be noisy")
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	spans := make(map[types.Object]span)
	for id, obj := range pass.TypesInfo.Defs {
		// Ignore identifiers that don't denote objects
		// (package names, symbolic variables such as t
		// in t := x.(type) of type switch headers).
		if obj != nil {
			growSpan(spans, obj, id.Pos(), id.End())
		}
	}
	for id, obj := range pass.TypesInfo.Uses {
		growSpan(spans, obj, id.Pos(), id.End())
	}
	for node, obj := range pass.TypesInfo.Implicits {
		// A type switch with a short variable declaration
		// such as t := x.(type) doesn't declare the symbolic
		// variable (t in the example) at the switch header;
		// instead a new variable t (with specific type) is
		// declared implicitly for each case. Such variables
		// are found in the types.Info.Implicits (not Defs)
		// map. Add them here, assuming they are declared at
		// the type cases' colon ":".
		if cc, ok := node.(*ast.CaseClause); ok {
			growSpan(spans, obj, cc.Colon, cc.Colon)
		}
	}

	nodeFilter := []ast.Node{
		(*ast.AssignStmt)(nil),
		(*ast.GenDecl)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		switch n := n.(type) {
		case *ast.AssignStmt:
			checkShadowAssignment(pass, spans, n)
		case *ast.GenDecl:
			checkShadowDecl(pass, spans, n)
		}
	})
	return nil, nil
}

// A span stores the minimum range of byte positions in the file in which a
// given variable (types.Object) is mentioned. It is lexically defined: it spans
// from the beginning of its first mention to the end of its last mention.
// A variable is considered shadowed (if strict is off) only if the
// shadowing variable is declared within the span of the shadowed variable.
// In other words, if a variable is shadowed but not used after the shadowed
// variable is declared, it is inconsequential and not worth complaining about.
// This simple check dramatically reduces the nuisance rate for the shadowing
// check, at least until something cleverer comes along.
//
// One wrinkle: A "naked return" is a silent use of a variable that the Span
// will not capture, but the compilers catch naked returns of shadowed
// variables so we don't need to.
//
// Cases this gets wrong (TODO):
// - If a for loop's continuation statement mentions a variable redeclared in
// the block, we should complain about it but don't.
// - A variable declared inside a function literal can falsely be identified
// as shadowing a variable in the outer function.
//
type span struct {
	min token.Pos
	max token.Pos
}

// contains reports whether the position is inside the span.
func (s span) contains(pos token.Pos) bool {
	return s.min <= pos && pos < s.max
}

// growSpan expands the span for the object to contain the source range [pos, end).
func growSpan(spans map[types.Object]span, obj types.Object, pos, end token.Pos) {
	if strict {
		return // No need
	}
	s, ok := spans[obj]
	if ok {
		if s.min > pos {
			s.min = pos
		}
		if s.max < end {
			s.max = end
		}
	} else {
		s = span{pos, end}
	}
	spans[obj] = s
}

// checkShadowAssignment checks for shadowing in a short variable declaration.
func checkShadowAssignment(pass *analysis.Pass, spans map[types.Object]span, a *ast.AssignStmt) {
	if a.Tok != token.DEFINE {
		return
	}
	if idiomaticShortRedecl(pass, a) {
		return
	}
	for _, expr := range a.Lhs {
		ident, ok := expr.(*ast.Ident)
		if !ok {
			pass.Reportf(expr.Pos(), "invalid AST: short variable declaration of non-identifier")
			return
		}
		checkShadowing(pass, spans, ident)
	}
}

// idiomaticShortRedecl reports whether this short declaration can be ignored for
// the purposes of shadowing, that is, that any redeclarations it contains are deliberate.
func idiomaticShortRedecl(pass *analysis.Pass, a *ast.AssignStmt) bool {
	// Don't complain about deliberate redeclarations of the form
	//	i := i
	// Such constructs are idiomatic in range loops to create a new variable
	// for each iteration. Another example is
	//	switch n := n.(type)
	if len(a.Rhs) != len(a.Lhs) {
		return false
	}
	// We know it's an assignment, so the LHS must be all identifiers. (We check anyway.)
	for i, expr := range a.Lhs {
		lhs, ok := expr.(*ast.Ident)
		if !ok {
			pass.Reportf(expr.Pos(), "invalid AST: short variable declaration of non-identifier")
			return true // Don't do any more processing.
		}
		switch rhs := a.Rhs[i].(type) {
		case *ast.Ident:
			if lhs.Name != rhs.Name {
				return false
			}
		case *ast.TypeAssertExpr:
			if id, ok := rhs.X.(*ast.Ident); ok {
				if lhs.Name != id.Name {
					return false
				}
			}
		default:
			return false
		}
	}
	return true
}

// idiomaticRedecl reports whether this declaration spec can be ignored for
// the purposes of shadowing, that is, that any redeclarations it contains are deliberate.
func idiomaticRedecl(d *ast.ValueSpec) bool {
	// Don't complain about deliberate redeclarations of the form
	//	var i, j = i, j
	if len(d.Names) != len(d.Values) {
		return false
	}
	for i, lhs := range d.Names {
		if rhs, ok := d.Values[i].(*ast.Ident); ok {
			if lhs.Name != rhs.Name {
				return false
			}
		}
	}
	return true
}

// checkShadowDecl checks for shadowing in a general variable declaration.
func checkShadowDecl(pass *analysis.Pass, spans map[types.Object]span, d *ast.GenDecl) {
	if d.Tok != token.VAR {
		return
	}
	for _, spec := range d.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			pass.Reportf(spec.Pos(), "invalid AST: var GenDecl not ValueSpec")
			return
		}
		// Don't complain about deliberate redeclarations of the form
		//	var i = i
		if idiomaticRedecl(valueSpec) {
			return
		}
		for _, ident := range valueSpec.Names {
			checkShadowing(pass, spans, ident)
		}
	}
}

// checkShadowing checks whether the identifier shadows an identifier in an outer scope.
func checkShadowing(pass *analysis.Pass, spans map[types.Object]span, ident *ast.Ident) {
	if ident.Name == "_" {
		// Can't shadow the blank identifier.
		return
	}
	obj := pass.TypesInfo.Defs[ident]
	if obj == nil {
		return
	}
	// obj.Parent.Parent is the surrounding scope. If we can find another declaration
	// starting from there, we have a shadowed identifier.
	_, shadowed := obj.Parent().Parent().LookupParent(obj.Name(), obj.Pos())
	if shadowed == nil {
		return
	}
	// Don't complain if it's shadowing a universe-declared identifier; that's fine.
	if shadowed.Parent() == types.Universe {
		return
	}
	if strict {
		// The shadowed identifier must appear before this one to be an instance of shadowing.
		if shadowed.Pos() > ident.Pos() {
			return
		}
	} else {
		// Don't complain if the span of validity of the shadowed identifier doesn't include
		// the shadowing identifier.
		span, ok := spans[shadowed]
		if !ok {
			pass.Reportf(ident.Pos(), "internal error: no range for %q", ident.Name)
			return
		}
		if !span.contains(ident.Pos()) {
			return
		}
	}
	// Don't complain if the types differ: that implies the programmer really wants two different things.
	if types.Identical(obj.Type(), shadowed.Type()) {
		line := pass.Fset.Position(shadowed.Pos()).Line
		pass.Reportf(ident.Pos(), "declaration of %q shadows declaration at line %d", obj.Name(), line)
	}
}
