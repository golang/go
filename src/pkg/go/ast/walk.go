// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import "fmt"


// A Visitor's Visit method is invoked for each node encountered by Walk.
// If Visit returns true, Walk is invoked for each of the node's children.
//
type Visitor interface {
	Visit(node interface{}) bool;
}


func walkIdent(v Visitor, x *Ident) {
	if x != nil {
		Walk(v, x);
	}
}


func walkCommentGroup(v Visitor, g *CommentGroup) {
	if g != nil {
		Walk(v, g);
	}
}


func walkFieldList(v Visitor, list []*Field) {
	for _, x := range list {
		Walk(v, x);
	}
}


func walkIdentList(v Visitor, list []*Ident) {
	for _, x := range list {
		walk(v, x);
	}
}


func walkExprList(v Visitor, list []Expr) {
	for _, x := range list {
		walk(v, x);
	}
}


func walkStmtList(v Visitor, list []Stmt) {
	for _, s := range list {
		walk(v, s);
	}
}


func walkBlockStmt(v Visitor, b *BlockStmt) {
	if b != nil {
		Walk(v, b);
	}
}


func walk(v Visitor, n Node) {
	if n != nil {
		Walk(v, n);
	}
}


// Walk recursively traverses an AST invokes v.Visit(n) for each
// node n encountered (starting with node). If v.Visit(n) returns
// true, Walk is invoked for each of the children of n.
//
func Walk(v Visitor, node interface{}) {
	if !v.Visit(node) {
		return;
	}

	// walk children
	switch n := node.(type) {
	// Comments and fields
	case *CommentGroup:
		for _, c := range n.List {
			Walk(v, c);
		}
		
	case *Field:
		walkCommentGroup(v, n.Doc);
		walkIdentList(v, n.Names);
		walk(v, n.Type);
		for _, x := range n.Tag {
			Walk(v, x);
		}
		walkCommentGroup(v, n.Comment);

	// Expressions
	case *StringList:
		for _, x := range n.Strings {
			Walk(v, x);
		}
		
	case *FuncLit:
		walk(v, n.Type);
		walkBlockStmt(v, n.Body);
		
	case *CompositeLit:
		walk(v, n.Type);
		walkExprList(v, n.Elts);
		
	case *ParenExpr:
		walk(v, n.X);
		
	case *SelectorExpr:
		walk(v, n.X);
		if n.Sel != nil {
			Walk(v, n.Sel);
		}
		
	case *IndexExpr:
		walk(v, n.X);
		walk(v, n.Index);
		walk(v, n.End);
		
	case *TypeAssertExpr:
		walk(v, n.X);
		walk(v, n.Type);
		
	case *CallExpr:
		walk(v, n.Fun);
		walkExprList(v, n.Args);
		
	case *StarExpr:
		walk(v, n.X);
		
	case *UnaryExpr:
		walk(v, n.X);
		
	case *BinaryExpr:
		walk(v, n.X);
		walk(v, n.Y);
		
	case *KeyValueExpr:
		walk(v, n.Key);
		walk(v, n.Value);

	// Types
	case *ArrayType:
		walk(v, n.Len);
		walk(v, n.Elt);
		
	case *StructType:
		walkFieldList(v, n.Fields);
		
	case *FuncType:
		walkFieldList(v, n.Params);
		walkFieldList(v, n.Results);
		
	case *InterfaceType:
		walkFieldList(v, n.Methods);
		
	case *MapType:
		walk(v, n.Key);
		walk(v, n.Value);
		
	case *ChanType:
		walk(v, n.Value);

	// Statements
	case *DeclStmt:
		walk(v, n.Decl);
		
	case *LabeledStmt:
		walkIdent(v, n.Label);
		walk(v, n.Stmt);
		
	case *ExprStmt:
		walk(v, n.X);
		
	case *IncDecStmt:
		walk(v, n.X);
		
	case *AssignStmt:
		walkExprList(v, n.Lhs);
		walkExprList(v, n.Rhs);
		
	case *GoStmt:
		if n.Call != nil {
			Walk(v, n.Call);
		}
		
	case *DeferStmt:
		if n.Call != nil {
			Walk(v, n.Call);
		}
		
	case *ReturnStmt:
		walkExprList(v, n.Results);
		
	case *BranchStmt:
		walkIdent(v, n.Label);
		
	case *BlockStmt:
		walkStmtList(v, n.List);
		
	case *IfStmt:
		walk(v, n.Init);
		walk(v, n.Cond);
		walkBlockStmt(v, n.Body);
		walk(v, n.Else);
		
	case *CaseClause:
		walkExprList(v, n.Values);
		walkStmtList(v, n.Body);
		
	case *SwitchStmt:
		walk(v, n.Init);
		walk(v, n.Tag);
		walkBlockStmt(v, n.Body);
		
	case *TypeCaseClause:
		walkExprList(v, n.Types);
		walkStmtList(v, n.Body);

	case *TypeSwitchStmt:
		walk(v, n.Init);
		walk(v, n.Assign);
		walkBlockStmt(v, n.Body);
		
	case *CommClause:
		walk(v, n.Lhs);
		walk(v, n.Rhs);
		walkStmtList(v, n.Body);
		
	case *SelectStmt:
		walkBlockStmt(v, n.Body);

	case *ForStmt:
		walk(v, n.Init);
		walk(v, n.Cond);
		walk(v, n.Post);
		walkBlockStmt(v, n.Body);
		
	case *RangeStmt:
		walk(v, n.Key);
		walk(v, n.Value);
		walk(v, n.X);
		walkBlockStmt(v, n.Body);
	
	// Declarations
	case *ImportSpec:
		walkCommentGroup(v, n.Doc);
		walkIdent(v, n.Name);
		for _, x := range n.Path {
			Walk(v, x);
		}
		walkCommentGroup(v, n.Comment);
		
		
	case *ValueSpec:
		walkCommentGroup(v, n.Doc);
		walkIdentList(v, n.Names);
		walk(v, n.Type);
		walkExprList(v, n.Values);
		walkCommentGroup(v, n.Comment);
		
	case *TypeSpec:
		walkCommentGroup(v, n.Doc);
		walkIdent(v, n.Name);
		walk(v, n.Type);
		walkCommentGroup(v, n.Comment);

	case *GenDecl:
		walkCommentGroup(v, n.Doc);
		for _, s := range n.Specs {
			Walk(v, s);
		}
		
	case *FuncDecl:
		walkCommentGroup(v, n.Doc);
		if n.Recv != nil {
			Walk(v, n.Recv);
		}
		walkIdent(v, n.Name);
		walk(v, n.Type);
		walkBlockStmt(v, n.Body);

	// Files and packages
	case *File:
		walkCommentGroup(v, n.Doc);
		walkIdent(v, n.Name);
		for _, d := range n.Decls {
			walk(v, d);
		}
		walkCommentGroup(v, n.Comments);

	case *Package:
		for _, f := range n.Files {
			Walk(v, f);
		}

	default:
		fmt.Printf("ast.Walk: unexpected type %T", n);
		panic();
	}
}
