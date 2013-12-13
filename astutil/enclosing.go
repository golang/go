// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

// This file defines utilities for working with source positions.

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
)

// PathEnclosingInterval returns the node that encloses the source
// interval [start, end), and all its ancestors up to the AST root.
//
// The definition of "enclosing" used by this function considers
// additional whitespace abutting a node to be enclosed by it.
// In this example:
//
//              z := x + y // add them
//                   <-A->
//                  <----B----->
//
// the ast.BinaryExpr(+) node is considered to enclose interval B
// even though its [Pos()..End()) is actually only interval A.
// This behaviour makes user interfaces more tolerant of imperfect
// input.
//
// This function treats tokens as nodes, though they are not included
// in the result. e.g. PathEnclosingInterval("+") returns the
// enclosing ast.BinaryExpr("x + y").
//
// If start==end, the 1-char interval following start is used instead.
//
// The 'exact' result is true if the interval contains only path[0]
// and perhaps some adjacent whitespace.  It is false if the interval
// overlaps multiple children of path[0], or if it contains only
// interior whitespace of path[0].
// In this example:
//
//              z := x + y // add them
//                <--C-->     <---E-->
//                  ^
//                  D
//
// intervals C, D and E are inexact.  C is contained by the
// z-assignment statement, because it spans three of its children (:=,
// x, +).  So too is the 1-char interval D, because it contains only
// interior whitespace of the assignment.  E is considered interior
// whitespace of the BlockStmt containing the assignment.
//
// Precondition: [start, end) both lie within the same file as root.
// TODO(adonovan): return (nil, false) in this case and remove precond.
// Requires FileSet; see importer.tokenFileContainsPos.
//
// Postcondition: path is never nil; it always contains at least 'root'.
//
func PathEnclosingInterval(root *ast.File, start, end token.Pos) (path []ast.Node, exact bool) {
	// fmt.Printf("EnclosingInterval %d %d\n", start, end) // debugging

	// Precondition: node.[Pos..End) and adjoining whitespace contain [start, end).
	var visit func(node ast.Node) bool
	visit = func(node ast.Node) bool {
		path = append(path, node)

		nodePos := node.Pos()
		nodeEnd := node.End()

		// fmt.Printf("visit(%T, %d, %d)\n", node, nodePos, nodeEnd) // debugging

		// Intersect [start, end) with interval of node.
		if start < nodePos {
			start = nodePos
		}
		if end > nodeEnd {
			end = nodeEnd
		}

		// Find sole child that contains [start, end).
		children := childrenOf(node)
		l := len(children)
		for i, child := range children {
			// [childPos, childEnd) is unaugmented interval of child.
			childPos := child.Pos()
			childEnd := child.End()

			// [augPos, augEnd) is whitespace-augmented interval of child.
			augPos := childPos
			augEnd := childEnd
			if i > 0 {
				augPos = children[i-1].End() // start of preceding whitespace
			}
			if i < l-1 {
				nextChildPos := children[i+1].Pos()
				// Does [start, end) lie between child and next child?
				if start >= augEnd && end <= nextChildPos {
					return false // inexact match
				}
				augEnd = nextChildPos // end of following whitespace
			}

			// fmt.Printf("\tchild %d: [%d..%d)\tcontains interval [%d..%d)?\n",
			// 	i, augPos, augEnd, start, end) // debugging

			// Does augmented child strictly contain [start, end)?
			if augPos <= start && end <= augEnd {
				_, isToken := child.(tokenNode)
				return isToken || visit(child)
			}

			// Does [start, end) overlap multiple children?
			// i.e. left-augmented child contains start
			// but LR-augmented child does not contain end.
			if start < childEnd && end > augEnd {
				break
			}
		}

		// No single child contained [start, end),
		// so node is the result.  Is it exact?

		// (It's tempting to put this condition before the
		// child loop, but it gives the wrong result in the
		// case where a node (e.g. ExprStmt) and its sole
		// child have equal intervals.)
		if start == nodePos && end == nodeEnd {
			return true // exact match
		}

		return false // inexact: overlaps multiple children
	}

	if start > end {
		start, end = end, start
	}

	if start < root.End() && end > root.Pos() {
		if start == end {
			end = start + 1 // empty interval => interval of size 1
		}
		exact = visit(root)

		// Reverse the path:
		for i, l := 0, len(path); i < l/2; i++ {
			path[i], path[l-1-i] = path[l-1-i], path[i]
		}
	} else {
		// Selection lies within whitespace preceding the
		// first (or following the last) declaration in the file.
		// The result nonetheless always includes the ast.File.
		path = append(path, root)
	}

	return
}

// tokenNode is a dummy implementation of ast.Node for a single token.
// They are used transiently by PathEnclosingInterval but never escape
// this package.
//
type tokenNode struct {
	pos token.Pos
	end token.Pos
}

func (n tokenNode) Pos() token.Pos {
	return n.pos
}

func (n tokenNode) End() token.Pos {
	return n.end
}

func tok(pos token.Pos, len int) ast.Node {
	return tokenNode{pos, pos + token.Pos(len)}
}

// childrenOf returns the direct non-nil children of ast.Node n.
// It may include fake ast.Node implementations for bare tokens.
// it is not safe to call (e.g.) ast.Walk on such nodes.
//
func childrenOf(n ast.Node) []ast.Node {
	var children []ast.Node

	// First add nodes for all true subtrees.
	ast.Inspect(n, func(node ast.Node) bool {
		if node == n { // push n
			return true // recur
		}
		if node != nil { // push child
			children = append(children, node)
		}
		return false // no recursion
	})

	// Then add fake Nodes for bare tokens.
	switch n := n.(type) {
	case *ast.ArrayType:
		children = append(children,
			tok(n.Lbrack, len("[")),
			tok(n.Elt.End(), len("]")))

	case *ast.AssignStmt:
		children = append(children,
			tok(n.TokPos, len(n.Tok.String())))

	case *ast.BasicLit:
		children = append(children,
			tok(n.ValuePos, len(n.Value)))

	case *ast.BinaryExpr:
		children = append(children, tok(n.OpPos, len(n.Op.String())))

	case *ast.BlockStmt:
		children = append(children,
			tok(n.Lbrace, len("{")),
			tok(n.Rbrace, len("}")))

	case *ast.BranchStmt:
		children = append(children,
			tok(n.TokPos, len(n.Tok.String())))

	case *ast.CallExpr:
		children = append(children,
			tok(n.Lparen, len("(")),
			tok(n.Rparen, len(")")))
		if n.Ellipsis != 0 {
			children = append(children, tok(n.Ellipsis, len("...")))
		}

	case *ast.CaseClause:
		if n.List == nil {
			children = append(children,
				tok(n.Case, len("default")))
		} else {
			children = append(children,
				tok(n.Case, len("case")))
		}
		children = append(children, tok(n.Colon, len(":")))

	case *ast.ChanType:
		switch n.Dir {
		case ast.RECV:
			children = append(children, tok(n.Begin, len("<-chan")))
		case ast.SEND:
			children = append(children, tok(n.Begin, len("chan<-")))
		case ast.RECV | ast.SEND:
			children = append(children, tok(n.Begin, len("chan")))
		}

	case *ast.CommClause:
		if n.Comm == nil {
			children = append(children,
				tok(n.Case, len("default")))
		} else {
			children = append(children,
				tok(n.Case, len("case")))
		}
		children = append(children, tok(n.Colon, len(":")))

	case *ast.Comment:
		// nop

	case *ast.CommentGroup:
		// nop

	case *ast.CompositeLit:
		children = append(children,
			tok(n.Lbrace, len("{")),
			tok(n.Rbrace, len("{")))

	case *ast.DeclStmt:
		// nop

	case *ast.DeferStmt:
		children = append(children,
			tok(n.Defer, len("defer")))

	case *ast.Ellipsis:
		children = append(children,
			tok(n.Ellipsis, len("...")))

	case *ast.EmptyStmt:
		// nop

	case *ast.ExprStmt:
		// nop

	case *ast.Field:
		// TODO(adonovan): Field.{Doc,Comment,Tag}?

	case *ast.FieldList:
		children = append(children,
			tok(n.Opening, len("(")),
			tok(n.Closing, len(")")))

	case *ast.File:
		// TODO test: Doc
		children = append(children,
			tok(n.Package, len("package")))

	case *ast.ForStmt:
		children = append(children,
			tok(n.For, len("for")))

	case *ast.FuncDecl:
		// TODO(adonovan): FuncDecl.Comment?

		// Uniquely, FuncDecl breaks the invariant that
		// preorder traversal yields tokens in lexical order:
		// in fact, FuncDecl.Recv precedes FuncDecl.Type.Func.
		//
		// As a workaround, we inline the case for FuncType
		// here and order things correctly.
		//
		children = nil // discard ast.Walk(FuncDecl) info subtrees
		children = append(children, tok(n.Type.Func, len("func")))
		if n.Recv != nil {
			children = append(children, n.Recv)
		}
		children = append(children, n.Name)
		if n.Type.Params != nil {
			children = append(children, n.Type.Params)
		}
		if n.Type.Results != nil {
			children = append(children, n.Type.Results)
		}
		if n.Body != nil {
			children = append(children, n.Body)
		}

	case *ast.FuncLit:
		// nop

	case *ast.FuncType:
		if n.Func != 0 {
			children = append(children,
				tok(n.Func, len("func")))
		}

	case *ast.GenDecl:
		children = append(children,
			tok(n.TokPos, len(n.Tok.String())))
		if n.Lparen != 0 {
			children = append(children,
				tok(n.Lparen, len("(")),
				tok(n.Rparen, len(")")))
		}

	case *ast.GoStmt:
		children = append(children,
			tok(n.Go, len("go")))

	case *ast.Ident:
		children = append(children,
			tok(n.NamePos, len(n.Name)))

	case *ast.IfStmt:
		children = append(children,
			tok(n.If, len("if")))

	case *ast.ImportSpec:
		// TODO(adonovan): ImportSpec.{Doc,EndPos}?

	case *ast.IncDecStmt:
		children = append(children,
			tok(n.TokPos, len(n.Tok.String())))

	case *ast.IndexExpr:
		children = append(children,
			tok(n.Lbrack, len("{")),
			tok(n.Rbrack, len("}")))

	case *ast.InterfaceType:
		children = append(children,
			tok(n.Interface, len("interface")))

	case *ast.KeyValueExpr:
		children = append(children,
			tok(n.Colon, len(":")))

	case *ast.LabeledStmt:
		children = append(children,
			tok(n.Colon, len(":")))

	case *ast.MapType:
		children = append(children,
			tok(n.Map, len("map")))

	case *ast.ParenExpr:
		children = append(children,
			tok(n.Lparen, len("(")),
			tok(n.Rparen, len(")")))

	case *ast.RangeStmt:
		children = append(children,
			tok(n.For, len("for")),
			tok(n.TokPos, len(n.Tok.String())))

	case *ast.ReturnStmt:
		children = append(children,
			tok(n.Return, len("return")))

	case *ast.SelectStmt:
		children = append(children,
			tok(n.Select, len("select")))

	case *ast.SelectorExpr:
		// nop

	case *ast.SendStmt:
		children = append(children,
			tok(n.Arrow, len("<-")))

	case *ast.SliceExpr:
		children = append(children,
			tok(n.Lbrack, len("[")),
			tok(n.Rbrack, len("]")))

	case *ast.StarExpr:
		children = append(children, tok(n.Star, len("*")))

	case *ast.StructType:
		children = append(children, tok(n.Struct, len("struct")))

	case *ast.SwitchStmt:
		children = append(children, tok(n.Switch, len("switch")))

	case *ast.TypeAssertExpr:
		children = append(children,
			tok(n.Lparen-1, len(".")),
			tok(n.Lparen, len("(")),
			tok(n.Rparen, len(")")))

	case *ast.TypeSpec:
		// TODO(adonovan): TypeSpec.{Doc,Comment}?

	case *ast.TypeSwitchStmt:
		children = append(children, tok(n.Switch, len("switch")))

	case *ast.UnaryExpr:
		children = append(children, tok(n.OpPos, len(n.Op.String())))

	case *ast.ValueSpec:
		// TODO(adonovan): ValueSpec.{Doc,Comment}?

	default:
		// Includes *ast.BadDecl, *ast.BadExpr, *ast.BadStmt.
		panic(fmt.Sprintf("unexpected node type %T", n))
	}

	// TODO(adonovan): opt: merge the logic of ast.Inspect() into
	// the switch above so we can make interleaved callbacks for
	// both Nodes and Tokens in the right order and avoid the need
	// to sort.
	sort.Sort(byPos(children))

	return children
}

type byPos []ast.Node

func (sl byPos) Len() int {
	return len(sl)
}
func (sl byPos) Less(i, j int) bool {
	return sl[i].Pos() < sl[j].Pos()
}
func (sl byPos) Swap(i, j int) {
	sl[i], sl[j] = sl[j], sl[i]
}

// NodeDescription returns a description of the concrete type of n suitable
// for a user interface.
//
// TODO(adonovan): in some cases (e.g. Field, FieldList, Ident,
// StarExpr) we could be much more specific given the path to the AST
// root.  Perhaps we should do that.
//
func NodeDescription(n ast.Node) string {
	switch n := n.(type) {
	case *ast.ArrayType:
		return "array type"
	case *ast.AssignStmt:
		return "assignment"
	case *ast.BadDecl:
		return "bad declaration"
	case *ast.BadExpr:
		return "bad expression"
	case *ast.BadStmt:
		return "bad statement"
	case *ast.BasicLit:
		return "basic literal"
	case *ast.BinaryExpr:
		return fmt.Sprintf("binary %s operation", n.Op)
	case *ast.BlockStmt:
		return "block"
	case *ast.BranchStmt:
		switch n.Tok {
		case token.BREAK:
			return "break statement"
		case token.CONTINUE:
			return "continue statement"
		case token.GOTO:
			return "goto statement"
		case token.FALLTHROUGH:
			return "fall-through statement"
		}
	case *ast.CallExpr:
		return "function call (or conversion)"
	case *ast.CaseClause:
		return "case clause"
	case *ast.ChanType:
		return "channel type"
	case *ast.CommClause:
		return "communication clause"
	case *ast.Comment:
		return "comment"
	case *ast.CommentGroup:
		return "comment group"
	case *ast.CompositeLit:
		return "composite literal"
	case *ast.DeclStmt:
		return NodeDescription(n.Decl) + " statement"
	case *ast.DeferStmt:
		return "defer statement"
	case *ast.Ellipsis:
		return "ellipsis"
	case *ast.EmptyStmt:
		return "empty statement"
	case *ast.ExprStmt:
		return "expression statement"
	case *ast.Field:
		// Can be any of these:
		// struct {x, y int}  -- struct field(s)
		// struct {T}         -- anon struct field
		// interface {I}      -- interface embedding
		// interface {f()}    -- interface method
		// func (A) func(B) C -- receiver, param(s), result(s)
		return "field/method/parameter"
	case *ast.FieldList:
		return "field/method/parameter list"
	case *ast.File:
		return "source file"
	case *ast.ForStmt:
		return "for loop"
	case *ast.FuncDecl:
		return "function declaration"
	case *ast.FuncLit:
		return "function literal"
	case *ast.FuncType:
		return "function type"
	case *ast.GenDecl:
		switch n.Tok {
		case token.IMPORT:
			return "import declaration"
		case token.CONST:
			return "constant declaration"
		case token.TYPE:
			return "type declaration"
		case token.VAR:
			return "variable declaration"
		}
	case *ast.GoStmt:
		return "go statement"
	case *ast.Ident:
		return "identifier"
	case *ast.IfStmt:
		return "if statement"
	case *ast.ImportSpec:
		return "import specification"
	case *ast.IncDecStmt:
		if n.Tok == token.INC {
			return "increment statement"
		}
		return "decrement statement"
	case *ast.IndexExpr:
		return "index expression"
	case *ast.InterfaceType:
		return "interface type"
	case *ast.KeyValueExpr:
		return "key/value association"
	case *ast.LabeledStmt:
		return "statement label"
	case *ast.MapType:
		return "map type"
	case *ast.Package:
		return "package"
	case *ast.ParenExpr:
		return "parenthesized " + NodeDescription(n.X)
	case *ast.RangeStmt:
		return "range loop"
	case *ast.ReturnStmt:
		return "return statement"
	case *ast.SelectStmt:
		return "select statement"
	case *ast.SelectorExpr:
		return "selector"
	case *ast.SendStmt:
		return "channel send"
	case *ast.SliceExpr:
		return "slice expression"
	case *ast.StarExpr:
		return "*-operation" // load/store expr or pointer type
	case *ast.StructType:
		return "struct type"
	case *ast.SwitchStmt:
		return "switch statement"
	case *ast.TypeAssertExpr:
		return "type assertion"
	case *ast.TypeSpec:
		return "type specification"
	case *ast.TypeSwitchStmt:
		return "type switch"
	case *ast.UnaryExpr:
		return fmt.Sprintf("unary %s operation", n.Op)
	case *ast.ValueSpec:
		return "value specification"

	}
	panic(fmt.Sprintf("unexpected node type: %T", n))
}
