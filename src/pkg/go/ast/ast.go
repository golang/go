// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The AST package declares the types used to represent
// syntax trees for Go packages.
//
package ast

import (
	"go/token";
	"unicode";
	"utf8";
)


// ----------------------------------------------------------------------------
// Interfaces
//
// There are 3 main classes of nodes: Expressions and type nodes,
// statement nodes, and declaration nodes. The node names usually
// match the corresponding Go spec production names to which they
// correspond. The node fields correspond to the individual parts
// of the respective productions.
//
// All nodes contain position information marking the beginning of
// the corresponding source text segment; it is accessible via the
// Pos accessor method. Nodes may contain additional position info
// for language constructs where comments may be found between parts
// of the construct (typically any larger, parenthesized subpart).
// That position information is needed to properly position comments
// when printing the construct.


// All node types implement the Node interface.
type Node interface {
	// Pos returns the (beginning) position of the node.
	Pos() token.Position;
}


// All expression nodes implement the Expr interface.
type Expr interface {
	Node;
	exprNode();
}


// All statement nodes implement the Stmt interface.
type Stmt interface {
	Node;
	stmtNode();
}


// All declaration nodes implement the Decl interface.
type Decl interface {
	Node;
	declNode();
}


// ----------------------------------------------------------------------------
// Comments

// A Comment node represents a single //-style or /*-style comment.
type Comment struct {
	token.Position;	// beginning position of the comment
	Text		[]byte;	// comment text (excluding '\n' for //-style comments)
}


// A CommentGroup represents a sequence of comments
// with no other tokens and no empty lines between.
//
type CommentGroup struct {
	List	[]*Comment;
	Next	*CommentGroup;	// next comment group in source order
}


// ----------------------------------------------------------------------------
// Expressions and types

// A Field represents a Field declaration list in a struct type,
// a method list in an interface type, or a parameter/result declaration
// in a signature.
//
type Field struct {
	Doc	*CommentGroup;	// associated documentation; or nil
	Names	[]*Ident;	// field/method/parameter names; or nil if anonymous field
	Type	Expr;		// field/method/parameter type
	Tag	[]*BasicLit;	// field tag; or nil
	Comment	*CommentGroup;	// line comments; or nil
}


func (f *Field) Pos() token.Position {
	if len(f.Names) > 0 {
		return f.Names[0].Pos();
	}
	return f.Type.Pos();
}


// An expression is represented by a tree consisting of one
// or more of the following concrete expression nodes.
//
type (
	// A BadExpr node is a placeholder for expressions containing
	// syntax errors for which no correct expression nodes can be
	// created.
	//
	BadExpr	struct {
		token.Position;	// beginning position of bad expression
	};

	// An Ident node represents an identifier.
	Ident	struct {
		token.Position;	// identifier position
		Value		string;	// identifier string (e.g. foobar)
	};

	// An Ellipsis node stands for the "..." type in a
	// parameter list or the "..." length in an array type.
	//
	Ellipsis	struct {
		token.Position;	// position of "..."
	};

	// A BasicLit node represents a literal of basic type.
	BasicLit	struct {
		token.Position;	// literal position
		Kind		token.Token;	//  token.INT, token.FLOAT, token.CHAR, or token.STRING
		Value		[]byte;		// literal string; e.g. 42, 0x7f, 3.14, 1e-9, 'a', '\x7f', "foo" or `\m\n\o`
	};

	// A StringList node represents a sequence of adjacent string literals.
	// A single string literal (common case) is represented by a BasicLit
	// node; StringList nodes are used only if there are two or more string
	// literals in a sequence.
	//
	StringList	struct {
		Strings []*BasicLit;	// list of strings, len(Strings) > 1
	};

	// A FuncLit node represents a function literal.
	FuncLit	struct {
		Type	*FuncType;	// function type
		Body	*BlockStmt;	// function body
	};

	// A CompositeLit node represents a composite literal.
	//
	CompositeLit	struct {
		Type	Expr;		// literal type
		Lbrace	token.Position;	// position of "{"
		Elts	[]Expr;		// list of composite elements
		Rbrace	token.Position;	// position of "}"
	};

	// A ParenExpr node represents a parenthesized expression.
	ParenExpr	struct {
		token.Position;	// position of "("
		X		Expr;		// parenthesized expression
		Rparen		token.Position;	// position of ")"
	};

	// A SelectorExpr node represents an expression followed by a selector.
	SelectorExpr	struct {
		X	Expr;	// expression
		Sel	*Ident;	// field selector
	};

	// An IndexExpr node represents an expression followed by an index or slice.
	IndexExpr	struct {
		X	Expr;	// expression
		Index	Expr;	// index expression or beginning of slice range
		End	Expr;	// end of slice range; or nil
	};

	// A TypeAssertExpr node represents an expression followed by a
	// type assertion.
	//
	TypeAssertExpr	struct {
		X	Expr;	// expression
		Type	Expr;	// asserted type; nil means type switch X.(type)
	};

	// A CallExpr node represents an expression followed by an argument list.
	CallExpr	struct {
		Fun	Expr;		// function expression
		Lparen	token.Position;	// position of "("
		Args	[]Expr;		// function arguments
		Rparen	token.Position;	// positions of ")"
	};

	// A StarExpr node represents an expression of the form "*" Expression.
	// Semantically it could be a unary "*" expression, or a pointer type.
	StarExpr	struct {
		token.Position;	// position of "*"
		X		Expr;	// operand
	};

	// A UnaryExpr node represents a unary expression.
	// Unary "*" expressions are represented via StarExpr nodes.
	//
	UnaryExpr	struct {
		token.Position;	// position of Op
		Op		token.Token;	// operator
		X		Expr;		// operand
	};

	// A BinaryExpr node represents a binary expression.
	//
	BinaryExpr	struct {
		X	Expr;		// left operand
		OpPos	token.Position;	// position of Op
		Op	token.Token;	// operator
		Y	Expr;		// right operand
	};

	// A KeyValueExpr node represents (key : value) pairs
	// in composite literals.
	//
	KeyValueExpr	struct {
		Key	Expr;
		Colon	token.Position;	// position of ":"
		Value	Expr;
	};
)


// The direction of a channel type is indicated by one
// of the following constants.
//
type ChanDir int

const (
	SEND	ChanDir	= 1<<iota;
	RECV;
)


// A type is represented by a tree consisting of one
// or more of the following type-specific expression
// nodes.
//
type (
	// An ArrayType node represents an array or slice type.
	ArrayType	struct {
		token.Position;	// position of "["
		Len		Expr;	// Ellipsis node for [...]T array types, nil for slice types
		Elt		Expr;	// element type
	};

	// A StructType node represents a struct type.
	StructType	struct {
		token.Position;	// position of "struct" keyword
		Lbrace		token.Position;	// position of "{"
		Fields		[]*Field;	// list of field declarations
		Rbrace		token.Position;	// position of "}"
		Incomplete	bool;		// true if (source) fields are missing in the Fields list
	};

	// Pointer types are represented via StarExpr nodes.

	// A FuncType node represents a function type.
	FuncType	struct {
		token.Position;	// position of "func" keyword
		Params		[]*Field;	// (incoming) parameters
		Results		[]*Field;	// (outgoing) results
	};

	// An InterfaceType node represents an interface type.
	InterfaceType	struct {
		token.Position;	// position of "interface" keyword
		Lbrace		token.Position;	// position of "{"
		Methods		[]*Field;	// list of methods
		Rbrace		token.Position;	// position of "}"
		Incomplete	bool;		// true if (source) methods are missing in the Methods list
	};

	// A MapType node represents a map type.
	MapType	struct {
		token.Position;	// position of "map" keyword
		Key		Expr;
		Value		Expr;
	};

	// A ChanType node represents a channel type.
	ChanType	struct {
		token.Position;	// position of "chan" keyword or "<-" (whichever comes first)
		Dir		ChanDir;	// channel direction
		Value		Expr;		// value type
	};
)


// Pos() implementations for expression/type where the position
// corresponds to the position of a sub-node.
//
func (x *StringList) Pos() token.Position {
	return x.Strings[0].Pos();
}
func (x *FuncLit) Pos() token.Position {
	return x.Type.Pos();
}
func (x *CompositeLit) Pos() token.Position {
	return x.Type.Pos();
}
func (x *SelectorExpr) Pos() token.Position {
	return x.X.Pos();
}
func (x *IndexExpr) Pos() token.Position {
	return x.X.Pos();
}
func (x *TypeAssertExpr) Pos() token.Position {
	return x.X.Pos();
}
func (x *CallExpr) Pos() token.Position {
	return x.Fun.Pos();
}
func (x *BinaryExpr) Pos() token.Position {
	return x.X.Pos();
}
func (x *KeyValueExpr) Pos() token.Position {
	return x.Key.Pos();
}


// exprNode() ensures that only expression/type nodes can be
// assigned to an ExprNode.
func (x *BadExpr) exprNode() {}
func (x *Ident) exprNode() {}
func (x *Ellipsis) exprNode() {}
func (x *BasicLit) exprNode() {}
func (x *StringList) exprNode() {}
func (x *FuncLit) exprNode() {}
func (x *CompositeLit) exprNode() {}
func (x *ParenExpr) exprNode() {}
func (x *SelectorExpr) exprNode() {}
func (x *IndexExpr) exprNode() {}
func (x *TypeAssertExpr) exprNode() {}
func (x *CallExpr) exprNode() {}
func (x *StarExpr) exprNode() {}
func (x *UnaryExpr) exprNode() {}
func (x *BinaryExpr) exprNode() {}
func (x *KeyValueExpr) exprNode() {}

func (x *ArrayType) exprNode() {}
func (x *StructType) exprNode() {}
func (x *FuncType) exprNode() {}
func (x *InterfaceType) exprNode() {}
func (x *MapType) exprNode() {}
func (x *ChanType) exprNode() {}


// IsExported returns whether name is an exported Go symbol
// (i.e., whether it begins with an uppercase letter).
func IsExported(name string) bool {
	ch, _ := utf8.DecodeRuneInString(name);
	return unicode.IsUpper(ch);
}

// IsExported returns whether name is an exported Go symbol
// (i.e., whether it begins with an uppercase letter).
func (name *Ident) IsExported() bool {
	return IsExported(name.Value);
}

func (name *Ident) String() string {
	return name.Value;
}


// ----------------------------------------------------------------------------
// Statements

// A statement is represented by a tree consisting of one
// or more of the following concrete statement nodes.
//
type (
	// A BadStmt node is a placeholder for statements containing
	// syntax errors for which no correct statement nodes can be
	// created.
	//
	BadStmt	struct {
		token.Position;	// beginning position of bad statement
	};

	// A DeclStmt node represents a declaration in a statement list.
	DeclStmt	struct {
		Decl Decl;
	};

	// An EmptyStmt node represents an empty statement.
	// The "position" of the empty statement is the position
	// of the immediately preceeding semicolon.
	//
	EmptyStmt	struct {
		token.Position;	// position of preceeding ";"
	};

	// A LabeledStmt node represents a labeled statement.
	LabeledStmt	struct {
		Label	*Ident;
		Stmt	Stmt;
	};

	// An ExprStmt node represents a (stand-alone) expression
	// in a statement list.
	//
	ExprStmt	struct {
		X Expr;	// expression
	};

	// An IncDecStmt node represents an increment or decrement statement.
	IncDecStmt	struct {
		X	Expr;
		Tok	token.Token;	// INC or DEC
	};

	// An AssignStmt node represents an assignment or
	// a short variable declaration.
	AssignStmt	struct {
		Lhs	[]Expr;
		TokPos	token.Position;	// position of Tok
		Tok	token.Token;	// assignment token, DEFINE
		Rhs	[]Expr;
	};

	// A GoStmt node represents a go statement.
	GoStmt	struct {
		token.Position;	// position of "go" keyword
		Call		*CallExpr;
	};

	// A DeferStmt node represents a defer statement.
	DeferStmt	struct {
		token.Position;	// position of "defer" keyword
		Call		*CallExpr;
	};

	// A ReturnStmt node represents a return statement.
	ReturnStmt	struct {
		token.Position;	// position of "return" keyword
		Results		[]Expr;
	};

	// A BranchStmt node represents a break, continue, goto,
	// or fallthrough statement.
	//
	BranchStmt	struct {
		token.Position;	// position of Tok
		Tok		token.Token;	// keyword token (BREAK, CONTINUE, GOTO, FALLTHROUGH)
		Label		*Ident;
	};

	// A BlockStmt node represents a braced statement list.
	BlockStmt	struct {
		token.Position;	// position of "{"
		List		[]Stmt;
		Rbrace		token.Position;	// position of "}"
	};

	// An IfStmt node represents an if statement.
	IfStmt	struct {
		token.Position;	// position of "if" keyword
		Init		Stmt;
		Cond		Expr;
		Body		*BlockStmt;
		Else		Stmt;
	};

	// A CaseClause represents a case of an expression switch statement.
	CaseClause	struct {
		token.Position;	// position of "case" or "default" keyword
		Values		[]Expr;		// nil means default case
		Colon		token.Position;	// position of ":"
		Body		[]Stmt;		// statement list; or nil
	};

	// A SwitchStmt node represents an expression switch statement.
	SwitchStmt	struct {
		token.Position;	// position of "switch" keyword
		Init		Stmt;
		Tag		Expr;
		Body		*BlockStmt;	// CaseClauses only
	};

	// A TypeCaseClause represents a case of a type switch statement.
	TypeCaseClause	struct {
		token.Position;	// position of "case" or "default" keyword
		Types		[]Expr;		// nil means default case
		Colon		token.Position;	// position of ":"
		Body		[]Stmt;		// statement list; or nil
	};

	// An TypeSwitchStmt node represents a type switch statement.
	TypeSwitchStmt	struct {
		token.Position;	// position of "switch" keyword
		Init		Stmt;
		Assign		Stmt;		// x := y.(type)
		Body		*BlockStmt;	// TypeCaseClauses only
	};

	// A CommClause node represents a case of a select statement.
	CommClause	struct {
		token.Position;	// position of "case" or "default" keyword
		Tok		token.Token;	// ASSIGN or DEFINE (valid only if Lhs != nil)
		Lhs, Rhs	Expr;		// Rhs == nil means default case
		Colon		token.Position;	// position of ":"
		Body		[]Stmt;		// statement list; or nil
	};

	// An SelectStmt node represents a select statement.
	SelectStmt	struct {
		token.Position;	// position of "select" keyword
		Body		*BlockStmt;	// CommClauses only
	};

	// A ForStmt represents a for statement.
	ForStmt	struct {
		token.Position;	// position of "for" keyword
		Init		Stmt;
		Cond		Expr;
		Post		Stmt;
		Body		*BlockStmt;
	};

	// A RangeStmt represents a for statement with a range clause.
	RangeStmt	struct {
		token.Position;	// position of "for" keyword
		Key, Value	Expr;		// Value may be nil
		TokPos		token.Position;	// position of Tok
		Tok		token.Token;	// ASSIGN, DEFINE
		X		Expr;		// value to range over
		Body		*BlockStmt;
	};
)


// Pos() implementations for statement nodes where the position
// corresponds to the position of a sub-node.
//
func (s *DeclStmt) Pos() token.Position {
	return s.Decl.Pos();
}
func (s *LabeledStmt) Pos() token.Position {
	return s.Label.Pos();
}
func (s *ExprStmt) Pos() token.Position {
	return s.X.Pos();
}
func (s *IncDecStmt) Pos() token.Position {
	return s.X.Pos();
}
func (s *AssignStmt) Pos() token.Position {
	return s.Lhs[0].Pos();
}


// stmtNode() ensures that only statement nodes can be
// assigned to a StmtNode.
//
func (s *BadStmt) stmtNode() {}
func (s *DeclStmt) stmtNode() {}
func (s *EmptyStmt) stmtNode() {}
func (s *LabeledStmt) stmtNode() {}
func (s *ExprStmt) stmtNode() {}
func (s *IncDecStmt) stmtNode() {}
func (s *AssignStmt) stmtNode() {}
func (s *GoStmt) stmtNode() {}
func (s *DeferStmt) stmtNode() {}
func (s *ReturnStmt) stmtNode() {}
func (s *BranchStmt) stmtNode() {}
func (s *BlockStmt) stmtNode() {}
func (s *IfStmt) stmtNode() {}
func (s *CaseClause) stmtNode() {}
func (s *SwitchStmt) stmtNode() {}
func (s *TypeCaseClause) stmtNode() {}
func (s *TypeSwitchStmt) stmtNode() {}
func (s *CommClause) stmtNode() {}
func (s *SelectStmt) stmtNode() {}
func (s *ForStmt) stmtNode() {}
func (s *RangeStmt) stmtNode() {}


// ----------------------------------------------------------------------------
// Declarations

// A Spec node represents a single (non-parenthesized) import,
// constant, type, or variable declaration.
//
type (
	// The Spec type stands for any of *ImportSpec, *ValueSpec, and *TypeSpec.
	Spec	interface {
		Node;
		specNode();
	};

	// An ImportSpec node represents a single package import.
	ImportSpec	struct {
		Doc	*CommentGroup;	// associated documentation; or nil
		Name	*Ident;		// local package name (including "."); or nil
		Path	[]*BasicLit;	// package path
		Comment	*CommentGroup;	// line comments; or nil
	};

	// A ValueSpec node represents a constant or variable declaration
	// (ConstSpec or VarSpec production).
	ValueSpec	struct {
		Doc	*CommentGroup;	// associated documentation; or nil
		Names	[]*Ident;	// value names
		Type	Expr;		// value type; or nil
		Values	[]Expr;		// initial values; or nil
		Comment	*CommentGroup;	// line comments; or nil
	};

	// A TypeSpec node represents a type declaration (TypeSpec production).
	TypeSpec	struct {
		Doc	*CommentGroup;	// associated documentation; or nil
		Name	*Ident;		// type name
		Type	Expr;
		Comment	*CommentGroup;	// line comments; or nil
	};
)


// Pos() implementations for spec nodes.
//
func (s *ImportSpec) Pos() token.Position {
	if s.Name != nil {
		return s.Name.Pos();
	}
	return s.Path[0].Pos();
}

func (s *ValueSpec) Pos() token.Position {
	return s.Names[0].Pos();
}
func (s *TypeSpec) Pos() token.Position {
	return s.Name.Pos();
}


// specNode() ensures that only spec nodes can be
// assigned to a Spec.
//
func (s *ImportSpec) specNode() {}
func (s *ValueSpec) specNode() {}
func (s *TypeSpec) specNode() {}


// A declaration is represented by one of the following declaration nodes.
//
type (
	// A BadDecl node is a placeholder for declarations containing
	// syntax errors for which no correct declaration nodes can be
	// created.
	//
	BadDecl	struct {
		token.Position;	// beginning position of bad declaration
	};

	// A GenDecl node (generic declaration node) represents an import,
	// constant, type or variable declaration. A valid Lparen position
	// (Lparen.Line > 0) indicates a parenthesized declaration.
	//
	// Relationship between Tok value and Specs element type:
	//
	//	token.IMPORT  *ImportSpec
	//	token.CONST   *ValueSpec
	//	token.TYPE    *TypeSpec
	//	token.VAR     *ValueSpec
	//
	GenDecl	struct {
		Doc		*CommentGroup;	// associated documentation; or nil
		token.Position;	// position of Tok
		Tok		token.Token;	// IMPORT, CONST, TYPE, VAR
		Lparen		token.Position;	// position of '(', if any
		Specs		[]Spec;
		Rparen		token.Position;	// position of ')', if any
	};

	// A FuncDecl node represents a function declaration.
	FuncDecl	struct {
		Doc	*CommentGroup;	// associated documentation; or nil
		Recv	*Field;		// receiver (methods); or nil (functions)
		Name	*Ident;		// function/method name
		Type	*FuncType;	// position of Func keyword, parameters and results
		Body	*BlockStmt;	// function body; or nil (forward declaration)
	};
)


// The position of a FuncDecl node is the position of its function type.
func (d *FuncDecl) Pos() token.Position {
	return d.Type.Pos();
}


// declNode() ensures that only declaration nodes can be
// assigned to a DeclNode.
//
func (d *BadDecl) declNode() {}
func (d *GenDecl) declNode() {}
func (d *FuncDecl) declNode() {}


// ----------------------------------------------------------------------------
// Files and packages

// A File node represents a Go source file.
//
type File struct {
	Doc		*CommentGroup;	// associated documentation; or nil
	token.Position;	// position of "package" keyword
	Name		*Ident;		// package name
	Decls		[]Decl;		// top-level declarations
	Comments	*CommentGroup;	// list of all comments in the source file
}


// A Package node represents a set of source files
// collectively building a Go package.
//
type Package struct {
	Name	string;			// package name
	Path	string;			// package path
	Files	map[string]*File;	// path-relative filenames
}
