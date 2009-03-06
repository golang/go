// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"vector";
	"token";
)


type (
	Block struct;
	Expr interface;
	Decl interface;
)


// ----------------------------------------------------------------------------
// Support

func assert(pred bool) {
	if !pred {
		panic("assertion failed");
	}
}


// ----------------------------------------------------------------------------
// All nodes have a source position and a token.

type Node struct {
	Pos int;  // source position (< 0 => unknown position)
	Tok int;  // identifying token
}


// ----------------------------------------------------------------------------
// Expressions

const /* channel mode */ (
	FULL = iota;
	SEND;
	RECV;
)


type (
	ExprVisitor interface;
	Signature struct;

	Expr interface {
		Pos() int;
		Visit(v ExprVisitor);
	};
	
	BadExpr struct {
		Pos_ int;
	};

	Ident struct {
		Pos_ int;
		Str string;
	};

	BinaryExpr struct {
		Pos_ int;
		Tok int;
		X, Y Expr;
	};

	UnaryExpr struct {
		Pos_ int;
		Tok int;
		X Expr;
	};

	// TODO this should probably just be a list instead
	ConcatExpr struct {
		X, Y Expr;
	};

	BasicLit struct {
		Pos_ int;
		Tok int;
		Val []byte;
	};

	FunctionLit struct {
		Pos_ int;  // position of "func"
		Typ *Signature;
		Body *Block;
	};
	
	Group struct {
		Pos_ int;  // position of "("
		X Expr;
	};

	Selector struct {
		Pos_ int;  // position of "."
		X Expr;
		Sel *Ident;
	};

	TypeGuard struct {
		Pos_ int;  // position of "."
		X Expr;
		Typ Expr;
	};

	Index struct {
		Pos_ int;  // position of "["
		X, I Expr;
	};
	
	Call struct {
		Pos_ int;  // position of "(" or "{"
		Tok int;
		F, Args Expr
	};

	// Type literals are treated like expressions.
	Ellipsis struct {  // neither a type nor an expression
		Pos_ int;
	};
	
	TypeType struct {  // for type switches
		Pos_ int;  // position of "type"
	};

	ArrayType struct {
		Pos_ int;  // position of "["
		Len Expr;
		Elt Expr;
	};
	
	Field struct {
		Idents []*Ident;
		Typ Expr;
		Tag Expr;  // nil = no tag
	};

	StructType struct {
		Pos_ int;  // position of "struct"
		Fields []*Field;
		End int;  // position of "}", End == 0 if forward declaration
	};
	
	PointerType struct {
		Pos_ int;  // position of "*"
		Base Expr;
	};
	
	Signature struct {
		Params []*Field;
		Result []*Field;
	};

	FunctionType struct {
		Pos_ int;  // position of "func"
		Sig *Signature;
	};

	InterfaceType struct {
		Pos_ int;  // position of "interface"
		Methods []*Field;
		End int;  // position of "}", End == 0 if forward declaration
	};

	SliceType struct {
		Pos_ int;  // position of "["
	};
	
	MapType struct {
		Pos_ int;  // position of "map"
		Key Expr;
		Val Expr;
	};
	
	ChannelType struct {
		Pos_ int;  // position of "chan" or "<-"
		Mode int;
		Val Expr;
	};
)


type ExprVisitor interface {
	DoBadExpr(x *BadExpr);
	DoIdent(x *Ident);
	DoBinaryExpr(x *BinaryExpr);
	DoUnaryExpr(x *UnaryExpr);
	DoConcatExpr(x *ConcatExpr);
	DoBasicLit(x *BasicLit);
	DoFunctionLit(x *FunctionLit);
	DoGroup(x *Group);
	DoSelector(x *Selector);
	DoTypeGuard(x *TypeGuard);
	DoIndex(x *Index);
	DoCall(x *Call);
	
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


// TODO replace these with an embedded field
func (x *BadExpr) Pos() int { return x.Pos_; }
func (x *Ident) Pos() int { return x.Pos_; }
func (x *BinaryExpr) Pos() int { return x.Pos_; }
func (x *UnaryExpr) Pos() int { return x.Pos_; }
func (x *ConcatExpr) Pos() int { return x.X.Pos(); }
func (x *BasicLit) Pos() int { return x.Pos_; }
func (x *FunctionLit) Pos() int { return x.Pos_; }
func (x *Group) Pos() int { return x.Pos_; }
func (x *Selector) Pos() int { return x.Pos_; }
func (x *TypeGuard) Pos() int { return x.Pos_; }
func (x *Index) Pos() int { return x.Pos_; }
func (x *Call) Pos() int { return x.Pos_; }

func (x *Ellipsis) Pos() int { return x.Pos_; }
func (x *TypeType) Pos() int { return x.Pos_; }
func (x *ArrayType) Pos() int { return x.Pos_; }
func (x *StructType) Pos() int { return x.Pos_; }
func (x *PointerType) Pos() int { return x.Pos_; }
func (x *FunctionType) Pos() int { return x.Pos_; }
func (x *InterfaceType) Pos() int { return x.Pos_; }
func (x *SliceType) Pos() int { return x.Pos_; }
func (x *MapType) Pos() int { return x.Pos_; }
func (x *ChannelType) Pos() int { return x.Pos_; }


func (x *BadExpr) Visit(v ExprVisitor) { v.DoBadExpr(x); }
func (x *Ident) Visit(v ExprVisitor) { v.DoIdent(x); }
func (x *BinaryExpr) Visit(v ExprVisitor) { v.DoBinaryExpr(x); }
func (x *UnaryExpr) Visit(v ExprVisitor) { v.DoUnaryExpr(x); }
func (x *ConcatExpr) Visit(v ExprVisitor) { v.DoConcatExpr(x); }
func (x *BasicLit) Visit(v ExprVisitor) { v.DoBasicLit(x); }
func (x *FunctionLit) Visit(v ExprVisitor) { v.DoFunctionLit(x); }
func (x *Group) Visit(v ExprVisitor) { v.DoGroup(x); }
func (x *Selector) Visit(v ExprVisitor) { v.DoSelector(x); }
func (x *TypeGuard) Visit(v ExprVisitor) { v.DoTypeGuard(x); }
func (x *Index) Visit(v ExprVisitor) { v.DoIndex(x); }
func (x *Call) Visit(v ExprVisitor) { v.DoCall(x); }

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



// Length of a comma-separated expression list.
func ExprLen(x Expr) int {
	if x == nil {
		return 0;
	}
	n := 1;
	for {
		if p, ok := x.(*BinaryExpr); ok && p.Tok == token.COMMA {
			n++;
			x = p.Y;
		} else {
			break;
		}
	}
	return n;
}


func ExprAt(x Expr, i int) Expr {
	for j := 0; j < i; j++ {
		assert(x.(*BinaryExpr).Tok == token.COMMA);
		x = x.(*BinaryExpr).Y;
	}
	if t, is_binary := x.(*BinaryExpr); is_binary && t.Tok == token.COMMA {
		x = t.X;
	}
	return x;
}


// ----------------------------------------------------------------------------
// Blocks
//
// Syntactic constructs of the form:
//
//   "{" StatementList "}"
//   ":" StatementList

type Block struct {
	Node;
	List *vector.Vector;
	End int;  // position of closing "}" if present
}


func NewBlock(pos, tok int) *Block {
	assert(tok == token.LBRACE || tok == token.COLON);
	b := new(Block);
	b.Pos, b.Tok, b.List = pos, tok, vector.New(0);
	return b;
}


// ----------------------------------------------------------------------------
// Statements

type (
	StatVisitor interface;

	Stat interface {
		Visit(v StatVisitor);
	};
	
	BadStat struct {
		Pos int;
	};

	LabelDecl struct {
		Pos int;  // position of ":"
		Label *Ident;
	};

	DeclarationStat struct {
		Decl Decl;
	};

	ExpressionStat struct {
		Pos int;  // position of Tok
		Tok int;  // INC, DEC, RETURN, GO, DEFER
		Expr Expr;
	};

	CompositeStat struct {
		Body *Block;
	};

	IfStat struct {
		Pos int;  // position of "if"
		Init Stat;
		Cond Expr;
		Body *Block;
		Else Stat;
	};
	
	ForStat struct {
		Pos int;  // position of "for"
		Init Stat;
		Cond Expr;
		Post Stat;
		Body *Block;
	};

	CaseClause struct {
		Pos int;  // position for "case" or "default"
		Expr Expr;  // nil means default case
		Body *Block;
	};

	SwitchStat struct {
		Pos int;  // position of "switch"
		Init Stat;
		Tag Expr;
		Body *Block;
	};
	
	SelectStat struct {
		Pos int;  // position of "select"
		Body *Block;
	};
	
	ControlFlowStat struct {
		Pos int;  // position of Tok
		Tok int;  // BREAK, CONTINUE, GOTO, FALLTHROUGH
		Label *Ident;  // if any, or nil
	};
	
	EmptyStat struct {
		Pos int;  // position of ";"
	};
)


type StatVisitor interface {
	DoBadStat(s *BadStat);
	DoLabelDecl(s *LabelDecl);
	DoDeclarationStat(s *DeclarationStat);
	DoExpressionStat(s *ExpressionStat);
	DoCompositeStat(s *CompositeStat);
	DoIfStat(s *IfStat);
	DoForStat(s *ForStat);
	DoCaseClause(s *CaseClause);
	DoSwitchStat(s *SwitchStat);
	DoSelectStat(s *SelectStat);
	DoControlFlowStat(s *ControlFlowStat);
	DoEmptyStat(s *EmptyStat);
}


func (s *BadStat) Visit(v StatVisitor) { v.DoBadStat(s); }
func (s *LabelDecl) Visit(v StatVisitor) { v.DoLabelDecl(s); }
func (s *DeclarationStat) Visit(v StatVisitor) { v.DoDeclarationStat(s); }
func (s *ExpressionStat) Visit(v StatVisitor) { v.DoExpressionStat(s); }
func (s *CompositeStat) Visit(v StatVisitor) { v.DoCompositeStat(s); }
func (s *IfStat) Visit(v StatVisitor) { v.DoIfStat(s); }
func (s *ForStat) Visit(v StatVisitor) { v.DoForStat(s); }
func (s *CaseClause) Visit(v StatVisitor) { v.DoCaseClause(s); }
func (s *SwitchStat) Visit(v StatVisitor) { v.DoSwitchStat(s); }
func (s *SelectStat) Visit(v StatVisitor) { v.DoSelectStat(s); }
func (s *ControlFlowStat) Visit(v StatVisitor) { v.DoControlFlowStat(s); }
func (s *EmptyStat) Visit(v StatVisitor) { v.DoEmptyStat(s); }


// ----------------------------------------------------------------------------
// Declarations

type (
	DeclVisitor interface;

	Decl interface {
		Visit(v DeclVisitor);
	};
	
	BadDecl struct {
		Pos int;
	};

	ImportDecl struct {
		Pos int;  // if > 0: position of "import"
		Ident *Ident;
		Path Expr;
	};
	
	ConstDecl struct {
		Pos int;  // if > 0: position of "const"
		Idents []*Ident;
		Typ Expr;
		Vals Expr;
	};
	
	TypeDecl struct {
		Pos int;  // if > 0: position of "type"
		Ident *Ident;
		Typ Expr;
	};
	
	VarDecl struct {
		Pos int;  // if > 0: position of "var"
		Idents []*Ident;
		Typ Expr;
		Vals Expr;
	};

	FuncDecl struct {
		Pos_ int;  // position of "func"
		Recv *Field;
		Ident *Ident;
		Sig *Signature;
		Body *Block;
	};
	
	DeclList struct {
		Pos int;  // position of Tok
		Tok int;
		List []Decl;
		End int;
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

type Comment struct {
	Pos int;
	Text []byte;
}


type Program struct {
	Pos int;  // tok is token.PACKAGE
	Ident *Ident;
	Decls []Decl;
	Comments []*Comment;
}


func NewProgram(pos int) *Program {
	p := new(Program);
	p.Pos = pos;
	return p;
}
