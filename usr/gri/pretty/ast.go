// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"vector";
	"token";
	"scanner";
)


type (
	Block struct;
	Expr interface;
	Decl interface;
	ExprVisitor interface;
	Signature struct;
)


// TODO rename scanner.Location to scanner.Position, possibly factor out
type Position scanner.Location


// ----------------------------------------------------------------------------
// Comments

type Comment struct {
	Loc scanner.Location;
	EndLine int;  // the line where the comment ends
	Text []byte;
}


// A CommentGroup is a sequence of consequtive comments
// with no other tokens and no empty lines inbetween.
type CommentGroup []*Comment


// ----------------------------------------------------------------------------
// Expressions and types


// All expression nodes implement the Expr interface.
type Expr interface {
	// For a (dynamic) node type X, calling Visit with an expression
	// visitor v invokes the node-specific DoX function of the visitor.
	//
	Visit(v ExprVisitor);
	
	// Pos returns the (beginning) position of the expression.
	Pos() Position;
};


// An expression is represented by a tree consisting of one
// or several of the following concrete expression nodes.
//
type (
	// A BadExpr node is a placeholder node for expressions containing
	// syntax errors for which not correct expression tree can be created.
	//
	BadExpr struct {
		Pos_ Position;  // bad expression position
	};


	// An Ident node represents an identifier (identifier).
	Ident struct {
		Str string;  // identifier string (e.g. foobar)
		Pos_ Position;  // identifier position
	};


	// An basic literal is represented by a BasicLit node.
	BasicLit struct {
		Tok int;  // literal token
		Lit []byte;  // literal string
		Pos_ Position;  // literal string position
	};


	// A sequence of string literals (StringLit) is represented
	// by a StringLit node.
	//
	StringLit struct {
		Strings []*BasicLit;  // sequence of strings
	};


	// A function literal (FunctionLit) is represented by a FunctionLit node.
	FunctionLit struct {
		Typ *Signature;  // function signature
		Body *Block;  // function body
		Func Position;  // position of "func" keyword
	};


	// A composite literal (CompositeLit) is represented by a CompositeLit node.
	CompositeLit struct {
		Typ Expr;  // literal type
		Elts []Expr;  // list of composite elements
		Lbrace, Rbrace Position;  // positions of "{" and "}"
	};


	// A parenthesized expression is represented by a Group node.
	Group struct {
		X Expr;  // parenthesized expression
		Lparen, Rparen Position;  // positions of "(" and ")"
	};


	// A primary expression followed by a selector is represented
	// by a Selector node.
	//
	Selector struct {
		X Expr;  // primary expression
		Sel *Ident;  // field selector
		Period Position;  // position of "."
	};


	// A primary expression followed by an index is represented
	// by an Index node.
	//
	Index struct {
		X Expr;  // primary expression
		Index Expr;  // index expression
		Lbrack, Rbrack Position;  // positions of "[" and "]"
	};


	// A primary expression followed by a slice is represented
	// by a Slice node.
	//
	Slice struct {
		X Expr;  // primary expression
		Beg, End Expr;  // slice range
		Lbrack, Colon, Rbrack Position;  // positions of "[", ":", and "]"
	};


	// A primary expression followed by a type assertion is represented
	// by a TypeAssertion node.
	//
	TypeAssertion struct {
		X Expr;  // primary expression
		Typ Expr;  // asserted type
		Period, Lparen, Rparen Position;  // positions of ".", "(", and ")"
	};


	// A primary expression followed by an argument list is represented
	// by a Call node.
	//
	Call struct {
		Fun Expr;  // function expression
		Args []Expr;  // function arguments
		Lparen, Rparen Position;  // positions of "(" and ")"
	};


	// A unary expression (UnaryExpr) is represented by a UnaryExpr node.
	UnaryExpr struct {
		Op int;  // operator token
		X Expr;  // operand
		Pos_ Position;  // operator position
	};


	// A binary expression (BinaryExpr) is represented by a BinaryExpr node.
	BinaryExpr struct {
		Op int;  // operator token
		X, Y Expr;  // left and right operand
		Pos_ Position;  // operator position
	};
)


// The direction of a channel type is indicated by one
// of the following constants.
//
const /* channel direction */ (
	FULL = iota;
	SEND;
	RECV;
)


type (
	// Type literals are treated like expressions.
	Ellipsis struct {  // neither a type nor an expression
		Loc_ scanner.Location;
	};
	
	TypeType struct {  // for type switches
		Loc_ scanner.Location;  // location of "type"
	};

	ArrayType struct {
		Loc_ scanner.Location;  // location of "["
		Len Expr;
		Elt Expr;
	};
	
	Field struct {
		Names []*Ident;
		Typ Expr;
		Tag Expr;  // nil = no tag
		Comment CommentGroup;
	};

	StructType struct {
		Loc_ scanner.Location;  // location of "struct"
		Fields []*Field;
		End scanner.Location;  // location of "}"
	};
	
	PointerType struct {
		Loc_ scanner.Location;  // location of "*"
		Base Expr;
	};
	
	Signature struct {
		Params []*Field;
		Result []*Field;
	};

	FunctionType struct {
		Loc_ scanner.Location;  // location of "func"
		Sig *Signature;
	};

	InterfaceType struct {
		Loc_ scanner.Location;  // location of "interface"
		Methods []*Field;
		End scanner.Location;  // location of "}", End == 0 if forward declaration
	};

	SliceType struct {
		Loc_ scanner.Location;  // location of "["
	};
	
	MapType struct {
		Loc_ scanner.Location;  // location of "map"
		Key Expr;
		Val Expr;
	};
	
	ChannelType struct {
		Loc_ scanner.Location;  // location of "chan" or "<-"
		Dir int;
		Val Expr;
	};
)


type ExprVisitor interface {
	// Expressions
	DoBadExpr(x *BadExpr);
	DoIdent(x *Ident);
	DoBasicLit(x *BasicLit);
	DoStringLit(x *StringLit);
	DoFunctionLit(x *FunctionLit);
	DoCompositeLit(x *CompositeLit);
	DoGroup(x *Group);
	DoSelector(x *Selector);
	DoIndex(x *Index);
	DoSlice(x *Slice);
	DoTypeAssertion(x *TypeAssertion);
	DoCall(x *Call);
	DoUnaryExpr(x *UnaryExpr);
	DoBinaryExpr(x *BinaryExpr);

	// Types
	DoEllipsis(x *Ellipsis);
	DoTypeType(x *TypeType);
	DoArrayType(x *ArrayType);
	DoStructType(x *StructType);
	DoPointerType(x *PointerType);
	DoFunctionType(x *FunctionType);
	DoInterfaceType(x *InterfaceType);
	DoSliceType(x *SliceType);
	DoMapType(x *MapType);
	DoChannelType(x *ChannelType);
}


func (x *BadExpr) Pos() Position  { return x.Pos_; }
func (x *Ident) Pos() Position  { return x.Pos_; }
func (x *BasicLit) Pos() Position  { return x.Pos_; }
func (x *StringLit) Pos() Position  { return x.Strings[0].Pos(); }
func (x *FunctionLit) Pos() Position  { return x.Func; }
func (x *CompositeLit) Pos() Position  { return x.Typ.Pos(); }
func (x *Group) Pos() Position  { return x.Lparen; }
func (x *Selector) Pos() Position  { return x.X.Pos(); }
func (x *Index) Pos() Position  { return x.X.Pos(); }
func (x *Slice) Pos() Position  { return x.X.Pos(); }
func (x *TypeAssertion) Pos() Position  { return x.X.Pos(); }
func (x *Call) Pos() Position  { return x.Fun.Pos(); }
func (x *UnaryExpr) Pos() Position  { return x.Pos_; }
func (x *BinaryExpr) Pos() Position  { return x.X.Pos(); }

func (x *Ellipsis) Pos() Position { return x.Loc_; }
func (x *TypeType) Pos() Position { return x.Loc_; }
func (x *ArrayType) Pos() Position { return x.Loc_; }
func (x *StructType) Pos() Position { return x.Loc_; }
func (x *PointerType) Pos() Position { return x.Loc_; }
func (x *FunctionType) Pos() Position { return x.Loc_; }
func (x *InterfaceType) Pos() Position { return x.Loc_; }
func (x *SliceType) Pos() Position { return x.Loc_; }
func (x *MapType) Pos() Position { return x.Loc_; }
func (x *ChannelType) Pos() Position { return x.Loc_; }


func (x *BadExpr) Visit(v ExprVisitor) { v.DoBadExpr(x); }
func (x *Ident) Visit(v ExprVisitor) { v.DoIdent(x); }
func (x *BasicLit) Visit(v ExprVisitor) { v.DoBasicLit(x); }
func (x *StringLit) Visit(v ExprVisitor) { v.DoStringLit(x); }
func (x *FunctionLit) Visit(v ExprVisitor) { v.DoFunctionLit(x); }
func (x *CompositeLit) Visit(v ExprVisitor) { v.DoCompositeLit(x); }
func (x *Group) Visit(v ExprVisitor) { v.DoGroup(x); }
func (x *Selector) Visit(v ExprVisitor) { v.DoSelector(x); }
func (x *Index) Visit(v ExprVisitor) { v.DoIndex(x); }
func (x *Slice) Visit(v ExprVisitor) { v.DoSlice(x); }
func (x *TypeAssertion) Visit(v ExprVisitor) { v.DoTypeAssertion(x); }
func (x *Call) Visit(v ExprVisitor) { v.DoCall(x); }
func (x *UnaryExpr) Visit(v ExprVisitor) { v.DoUnaryExpr(x); }
func (x *BinaryExpr) Visit(v ExprVisitor) { v.DoBinaryExpr(x); }

func (x *Ellipsis) Visit(v ExprVisitor) { v.DoEllipsis(x); }
func (x *TypeType) Visit(v ExprVisitor) { v.DoTypeType(x); }
func (x *ArrayType) Visit(v ExprVisitor) { v.DoArrayType(x); }
func (x *StructType) Visit(v ExprVisitor) { v.DoStructType(x); }
func (x *PointerType) Visit(v ExprVisitor) { v.DoPointerType(x); }
func (x *FunctionType) Visit(v ExprVisitor) { v.DoFunctionType(x); }
func (x *InterfaceType) Visit(v ExprVisitor) { v.DoInterfaceType(x); }
func (x *SliceType) Visit(v ExprVisitor) { v.DoSliceType(x); }
func (x *MapType) Visit(v ExprVisitor) { v.DoMapType(x); }
func (x *ChannelType) Visit(v ExprVisitor) { v.DoChannelType(x); }


// ----------------------------------------------------------------------------
// Blocks
//
// Syntactic constructs of the form:
//
//   "{" StatementList "}"
//   ":" StatementList

type Block struct {
	Loc scanner.Location;
	Tok int;
	List *vector.Vector;
	End scanner.Location;  // location of closing "}" if present
}


func NewBlock(loc scanner.Location, tok int) *Block {
	if tok != token.LBRACE && tok != token.COLON {
		panic();
	}
	var end scanner.Location;
	return &Block{loc, tok, vector.New(0), end};
}


// ----------------------------------------------------------------------------
// Statements

type (
	StatVisitor interface;

	Stat interface {
		Visit(v StatVisitor);
	};
	
	BadStat struct {
		Loc scanner.Location;
	};

	LabeledStat struct {
		Loc scanner.Location;  // location of ":"
		Label *Ident;
		Stat Stat;
	};

	DeclarationStat struct {
		Decl Decl;
	};

	ExpressionStat struct {
		Loc scanner.Location;  // location of Tok
		Tok int;  // GO, DEFER
		Expr Expr;
	};

	AssignmentStat struct {
		Loc scanner.Location;  // location of Tok
		Tok int;  // assignment token
		Lhs, Rhs Expr;
	};

	TupleAssignStat struct {
		Loc scanner.Location;  // location of Tok
		Tok int;  // assignment token
		Lhs, Rhs []Expr;
	};

	IncDecStat struct {
		Loc scanner.Location;  // location of '++' or '--'
		Tok int;  // token.INC or token.DEC
		Expr Expr;
	};

	CompositeStat struct {
		Body *Block;
	};

	IfStat struct {
		Loc scanner.Location;  // location of "if"
		Init Stat;
		Cond Expr;
		Body *Block;
		Else Stat;
	};
	
	RangeClause struct {  // appears only as Init stat in a ForStat
		Loc scanner.Location;  // location of "=" or ":="
		Tok int;  // token.ASSIGN or token.DEFINE
		Lhs []Expr;
		Rhs Expr;
	};

	ForStat struct {
		Loc scanner.Location;  // location of "for"
		Init Stat;
		Cond Expr;
		Post Stat;
		Body *Block;
	};

	TypeSwitchClause struct {  // appears only as Init stat in a SwitchStat
		Loc scanner.Location;  // location of ":="
		Lhs *Ident;
		Rhs Expr;
	};

	CaseClause struct {
		Loc scanner.Location;  // location of "case" or "default"
		Values []Expr;  // nil means default case
		Body *Block;
	};

	SwitchStat struct {
		Loc scanner.Location;  // location of "switch"
		Init Stat;
		Tag Expr;
		Body *Block;
	};

	CommClause struct {
		Loc scanner.Location;  // location of "case" or "default"
		Tok int;  // token.ASSIGN, token.DEFINE (valid only if Lhs != nil)
		Lhs, Rhs Expr;  // Rhs == nil means default case
		Body *Block;
	};

	SelectStat struct {
		Loc scanner.Location;  // location of "select"
		Body *Block;
	};
	
	ControlFlowStat struct {
		Loc scanner.Location;  // location of Tok
		Tok int;  // BREAK, CONTINUE, GOTO, FALLTHROUGH
		Label *Ident;  // if any, or nil
	};
	
	ReturnStat struct {
		Loc scanner.Location;  // location of "return"
		Results []Expr;
	};
	
	EmptyStat struct {
		Loc scanner.Location;  // location of ";"
	};
)


type StatVisitor interface {
	DoBadStat(s *BadStat);
	DoLabeledStat(s *LabeledStat);
	DoDeclarationStat(s *DeclarationStat);
	DoExpressionStat(s *ExpressionStat);
	DoAssignmentStat(s *AssignmentStat);
	DoTupleAssignStat(s *TupleAssignStat);
	DoIncDecStat(s *IncDecStat);
	DoCompositeStat(s *CompositeStat);
	DoIfStat(s *IfStat);
	DoRangeClause(s *RangeClause);
	DoForStat(s *ForStat);
	DoTypeSwitchClause(s *TypeSwitchClause);
	DoCaseClause(s *CaseClause);
	DoSwitchStat(s *SwitchStat);
	DoCommClause(s *CommClause);
	DoSelectStat(s *SelectStat);
	DoControlFlowStat(s *ControlFlowStat);
	DoReturnStat(s *ReturnStat);
	DoEmptyStat(s *EmptyStat);
}


func (s *BadStat) Visit(v StatVisitor) { v.DoBadStat(s); }
func (s *LabeledStat) Visit(v StatVisitor) { v.DoLabeledStat(s); }
func (s *DeclarationStat) Visit(v StatVisitor) { v.DoDeclarationStat(s); }
func (s *ExpressionStat) Visit(v StatVisitor) { v.DoExpressionStat(s); }
func (s *AssignmentStat) Visit(v StatVisitor) { v.DoAssignmentStat(s); }
func (s *TupleAssignStat) Visit(v StatVisitor) { v.DoTupleAssignStat(s); }
func (s *IncDecStat) Visit(v StatVisitor) { v.DoIncDecStat(s); }
func (s *CompositeStat) Visit(v StatVisitor) { v.DoCompositeStat(s); }
func (s *IfStat) Visit(v StatVisitor) { v.DoIfStat(s); }
func (s *RangeClause) Visit(v StatVisitor) { v.DoRangeClause(s); }
func (s *ForStat) Visit(v StatVisitor) { v.DoForStat(s); }
func (s *TypeSwitchClause) Visit(v StatVisitor) { v.DoTypeSwitchClause(s); }
func (s *CaseClause) Visit(v StatVisitor) { v.DoCaseClause(s); }
func (s *SwitchStat) Visit(v StatVisitor) { v.DoSwitchStat(s); }
func (s *CommClause) Visit(v StatVisitor) { v.DoCommClause(s); }
func (s *SelectStat) Visit(v StatVisitor) { v.DoSelectStat(s); }
func (s *ControlFlowStat) Visit(v StatVisitor) { v.DoControlFlowStat(s); }
func (s *ReturnStat) Visit(v StatVisitor) { v.DoReturnStat(s); }
func (s *EmptyStat) Visit(v StatVisitor) { v.DoEmptyStat(s); }


// ----------------------------------------------------------------------------
// Declarations

type (
	DeclVisitor interface;

	Decl interface {
		Visit(v DeclVisitor);
	};
	
	BadDecl struct {
		Loc scanner.Location;
	};

	ImportDecl struct {
		Loc scanner.Location;  // if > 0: position of "import"
		Name *Ident;
		Path Expr;
	};
	
	ConstDecl struct {
		Loc scanner.Location;  // if > 0: position of "const"
		Names []*Ident;
		Typ Expr;
		Values []Expr;
		Comment CommentGroup;
	};
	
	TypeDecl struct {
		Loc scanner.Location;  // if > 0: position of "type"
		Name *Ident;
		Typ Expr;
		Comment CommentGroup;
	};
	
	VarDecl struct {
		Loc scanner.Location;  // if > 0: position of "var"
		Names []*Ident;
		Typ Expr;
		Values []Expr;
		Comment CommentGroup;
	};

	FuncDecl struct {
		Loc scanner.Location;  // location of "func"
		Recv *Field;
		Name *Ident;
		Sig *Signature;
		Body *Block;
		Comment CommentGroup;
	};
	
	DeclList struct {
		Loc scanner.Location;  // location of Tok
		Tok int;
		List []Decl;
		End scanner.Location;
	};
)


type DeclVisitor interface {
	DoBadDecl(d *BadDecl);
	DoImportDecl(d *ImportDecl);
	DoConstDecl(d *ConstDecl);
	DoTypeDecl(d *TypeDecl);
	DoVarDecl(d *VarDecl);
	DoFuncDecl(d *FuncDecl);
	DoDeclList(d *DeclList);
}


func (d *BadDecl) Visit(v DeclVisitor) { v.DoBadDecl(d); }
func (d *ImportDecl) Visit(v DeclVisitor) { v.DoImportDecl(d); }
func (d *ConstDecl) Visit(v DeclVisitor) { v.DoConstDecl(d); }
func (d *TypeDecl) Visit(v DeclVisitor) { v.DoTypeDecl(d); }
func (d *VarDecl) Visit(v DeclVisitor) { v.DoVarDecl(d); }
func (d *FuncDecl) Visit(v DeclVisitor) { v.DoFuncDecl(d); }
func (d *DeclList) Visit(v DeclVisitor) { v.DoDeclList(d); }


// ----------------------------------------------------------------------------
// Program

// TODO rename to Package
type Program struct {
	Loc scanner.Location;  // tok is token.PACKAGE
	Name *Ident;
	Decls []Decl;
	Comment CommentGroup;
	Comments []CommentGroup;
}


func NewProgram(loc scanner.Location) *Program {
	p := new(Program);
	p.Loc = loc;
	return p;
}
