// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unreachable

// TODO(adonovan): use the new cfg package, which is more precise.

import (
	_ "embed"
	"go/ast"
	"go/token"
	"log"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/refactor"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:             "unreachable",
	Doc:              analysisinternal.MustExtractDoc(doc, "unreachable"),
	URL:              "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/unreachable",
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	RunDespiteErrors: true,
	Run:              run,
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.FuncDecl)(nil),
		(*ast.FuncLit)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		var body *ast.BlockStmt
		switch n := n.(type) {
		case *ast.FuncDecl:
			body = n.Body
		case *ast.FuncLit:
			body = n.Body
		}
		if body == nil {
			return
		}
		d := &deadState{
			pass:     pass,
			hasBreak: make(map[ast.Stmt]bool),
			hasGoto:  make(map[string]bool),
			labels:   make(map[string]ast.Stmt),
		}
		d.findLabels(body)
		d.reachable = true
		d.findDead(body)
	})
	return nil, nil
}

type deadState struct {
	pass        *analysis.Pass
	hasBreak    map[ast.Stmt]bool
	hasGoto     map[string]bool
	labels      map[string]ast.Stmt
	breakTarget ast.Stmt

	reachable bool
}

// findLabels gathers information about the labels defined and used by stmt
// and about which statements break, whether a label is involved or not.
func (d *deadState) findLabels(stmt ast.Stmt) {
	switch x := stmt.(type) {
	default:
		log.Fatalf("%s: internal error in findLabels: unexpected statement %T", d.pass.Fset.Position(x.Pos()), x)

	case *ast.AssignStmt,
		*ast.BadStmt,
		*ast.DeclStmt,
		*ast.DeferStmt,
		*ast.EmptyStmt,
		*ast.ExprStmt,
		*ast.GoStmt,
		*ast.IncDecStmt,
		*ast.ReturnStmt,
		*ast.SendStmt:
		// no statements inside

	case *ast.BlockStmt:
		for _, stmt := range x.List {
			d.findLabels(stmt)
		}

	case *ast.BranchStmt:
		switch x.Tok {
		case token.GOTO:
			if x.Label != nil {
				d.hasGoto[x.Label.Name] = true
			}

		case token.BREAK:
			stmt := d.breakTarget
			if x.Label != nil {
				stmt = d.labels[x.Label.Name]
			}
			if stmt != nil {
				d.hasBreak[stmt] = true
			}
		}

	case *ast.IfStmt:
		d.findLabels(x.Body)
		if x.Else != nil {
			d.findLabels(x.Else)
		}

	case *ast.LabeledStmt:
		d.labels[x.Label.Name] = x.Stmt
		d.findLabels(x.Stmt)

	// These cases are all the same, but the x.Body only works
	// when the specific type of x is known, so the cases cannot
	// be merged.
	case *ast.ForStmt:
		outer := d.breakTarget
		d.breakTarget = x
		d.findLabels(x.Body)
		d.breakTarget = outer

	case *ast.RangeStmt:
		outer := d.breakTarget
		d.breakTarget = x
		d.findLabels(x.Body)
		d.breakTarget = outer

	case *ast.SelectStmt:
		outer := d.breakTarget
		d.breakTarget = x
		d.findLabels(x.Body)
		d.breakTarget = outer

	case *ast.SwitchStmt:
		outer := d.breakTarget
		d.breakTarget = x
		d.findLabels(x.Body)
		d.breakTarget = outer

	case *ast.TypeSwitchStmt:
		outer := d.breakTarget
		d.breakTarget = x
		d.findLabels(x.Body)
		d.breakTarget = outer

	case *ast.CommClause:
		for _, stmt := range x.Body {
			d.findLabels(stmt)
		}

	case *ast.CaseClause:
		for _, stmt := range x.Body {
			d.findLabels(stmt)
		}
	}
}

// findDead walks the statement looking for dead code.
// If d.reachable is false on entry, stmt itself is dead.
// When findDead returns, d.reachable tells whether the
// statement following stmt is reachable.
func (d *deadState) findDead(stmt ast.Stmt) {
	// Is this a labeled goto target?
	// If so, assume it is reachable due to the goto.
	// This is slightly conservative, in that we don't
	// check that the goto is reachable, so
	//	L: goto L
	// will not provoke a warning.
	// But it's good enough.
	if x, isLabel := stmt.(*ast.LabeledStmt); isLabel && d.hasGoto[x.Label.Name] {
		d.reachable = true
	}

	if !d.reachable {
		switch stmt.(type) {
		case *ast.EmptyStmt:
			// do not warn about unreachable empty statements
		default:
			var (
				inspect    = d.pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
				curStmt, _ = inspect.Root().FindNode(stmt)
				tokFile    = d.pass.Fset.File(stmt.Pos())
			)
			// (This call to pass.Report is a frequent source
			// of diagnostics beyond EOF in a truncated file;
			// see #71659.)
			d.pass.Report(analysis.Diagnostic{
				Pos:     stmt.Pos(),
				End:     stmt.End(),
				Message: "unreachable code",
				SuggestedFixes: []analysis.SuggestedFix{{
					Message:   "Remove",
					TextEdits: refactor.DeleteStmt(tokFile, curStmt),
				}},
			})
			d.reachable = true // silence error about next statement
		}
	}

	switch x := stmt.(type) {
	default:
		log.Fatalf("%s: internal error in findDead: unexpected statement %T", d.pass.Fset.Position(x.Pos()), x)

	case *ast.AssignStmt,
		*ast.BadStmt,
		*ast.DeclStmt,
		*ast.DeferStmt,
		*ast.EmptyStmt,
		*ast.GoStmt,
		*ast.IncDecStmt,
		*ast.SendStmt:
		// no control flow

	case *ast.BlockStmt:
		for _, stmt := range x.List {
			d.findDead(stmt)
		}

	case *ast.BranchStmt:
		switch x.Tok {
		case token.BREAK, token.GOTO, token.FALLTHROUGH:
			d.reachable = false
		case token.CONTINUE:
			// NOTE: We accept "continue" statements as terminating.
			// They are not necessary in the spec definition of terminating,
			// because a continue statement cannot be the final statement
			// before a return. But for the more general problem of syntactically
			// identifying dead code, continue redirects control flow just
			// like the other terminating statements.
			d.reachable = false
		}

	case *ast.ExprStmt:
		// Call to panic?
		call, ok := x.X.(*ast.CallExpr)
		if ok {
			name, ok := call.Fun.(*ast.Ident)
			if ok && name.Name == "panic" && name.Obj == nil {
				d.reachable = false
			}
		}

	case *ast.ForStmt:
		d.findDead(x.Body)
		d.reachable = x.Cond != nil || d.hasBreak[x]

	case *ast.IfStmt:
		d.findDead(x.Body)
		if x.Else != nil {
			r := d.reachable
			d.reachable = true
			d.findDead(x.Else)
			d.reachable = d.reachable || r
		} else {
			// might not have executed if statement
			d.reachable = true
		}

	case *ast.LabeledStmt:
		d.findDead(x.Stmt)

	case *ast.RangeStmt:
		d.findDead(x.Body)
		d.reachable = true

	case *ast.ReturnStmt:
		d.reachable = false

	case *ast.SelectStmt:
		// NOTE: Unlike switch and type switch below, we don't care
		// whether a select has a default, because a select without a
		// default blocks until one of the cases can run. That's different
		// from a switch without a default, which behaves like it has
		// a default with an empty body.
		anyReachable := false
		for _, comm := range x.Body.List {
			d.reachable = true
			for _, stmt := range comm.(*ast.CommClause).Body {
				d.findDead(stmt)
			}
			anyReachable = anyReachable || d.reachable
		}
		d.reachable = anyReachable || d.hasBreak[x]

	case *ast.SwitchStmt:
		anyReachable := false
		hasDefault := false
		for _, cas := range x.Body.List {
			cc := cas.(*ast.CaseClause)
			if cc.List == nil {
				hasDefault = true
			}
			d.reachable = true
			for _, stmt := range cc.Body {
				d.findDead(stmt)
			}
			anyReachable = anyReachable || d.reachable
		}
		d.reachable = anyReachable || d.hasBreak[x] || !hasDefault

	case *ast.TypeSwitchStmt:
		anyReachable := false
		hasDefault := false
		for _, cas := range x.Body.List {
			cc := cas.(*ast.CaseClause)
			if cc.List == nil {
				hasDefault = true
			}
			d.reachable = true
			for _, stmt := range cc.Body {
				d.findDead(stmt)
			}
			anyReachable = anyReachable || d.reachable
		}
		d.reachable = anyReachable || d.hasBreak[x] || !hasDefault
	}
}
