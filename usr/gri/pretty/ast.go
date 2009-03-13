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
)


// ----------------------------------------------------------------------------
// Support

func assert(pred bool) {
	if !pred {
		panic("assertion failed");
	}
}


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
		Loc() scanner.Location;
		Visit(v ExprVisitor);
	};
	
	BadExpr struct {
		Loc_ scanner.Location;
	};

	Ident struct {
		Loc_ scanner.Location;
		Str string;
	};

	BinaryExpr struct {
		Loc_ scanner.Location;
		Tok int;
		X, Y Expr;
	};

	UnaryExpr struct {
		Loc_ scanner.Location;
		Tok int;
		X Expr;
	};

	// TODO this should probably just be a list instead
	ConcatExpr struct {
		X, Y Expr;
	};

	BasicLit struct {
		Loc_ scanner.Location;
		Tok int;
		Val []byte;
	};

	FunctionLit struct {
		Loc_ scanner.Location;  // location of "func"
		Typ *Signature;
		Body *Block;
	};
	
	Group struct {
		Loc_ scanner.Location;  // location of "("
		X Expr;
	};

	Selector struct {
		Loc_ scanner.Location;  // location of "."
		X Expr;
		Sel *Ident;
	};

	TypeGuard struct {
		Loc_ scanner.Location;  // location of "."
		X Expr;
		Typ Expr;
	};

	Index struct {
		Loc_ scanner.Location;  // location of "["
		X, I Expr;
	};
	
	Call struct {
		Loc_ scanner.Location;  // location of "(" or "{"
		Tok int;
		F, Args Expr
	};

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
		Idents []*Ident;
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
func (x *BadExpr) Loc() scanner.Location { return x.Loc_; }
func (x *Ident) Loc() scanner.Location { return x.Loc_; }
func (x *BinaryExpr) Loc() scanner.Location { return x.Loc_; }
func (x *UnaryExpr) Loc() scanner.Location { return x.Loc_; }
func (x *ConcatExpr) Loc() scanner.Location { return x.X.Loc(); }
func (x *BasicLit) Loc() scanner.Location { return x.Loc_; }
func (x *FunctionLit) Loc() scanner.Location { return x.Loc_; }
func (x *Group) Loc() scanner.Location { return x.Loc_; }
func (x *Selector) Loc() scanner.Location { return x.Loc_; }
func (x *TypeGuard) Loc() scanner.Location { return x.Loc_; }
func (x *Index) Loc() scanner.Location { return x.Loc_; }
func (x *Call) Loc() scanner.Location { return x.Loc_; }

func (x *Ellipsis) Loc() scanner.Location { return x.Loc_; }
func (x *TypeType) Loc() scanner.Location { return x.Loc_; }
func (x *ArrayType) Loc() scanner.Location { return x.Loc_; }
func (x *StructType) Loc() scanner.Location { return x.Loc_; }
func (x *PointerType) Loc() scanner.Location { return x.Loc_; }
func (x *FunctionType) Loc() scanner.Location { return x.Loc_; }
func (x *InterfaceType) Loc() scanner.Location { return x.Loc_; }
func (x *SliceType) Loc() scanner.Location { return x.Loc_; }
func (x *MapType) Loc() scanner.Location { return x.Loc_; }
func (x *ChannelType) Loc() scanner.Location { return x.Loc_; }


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
	Loc scanner.Location;
	Tok int;
	List *vector.Vector;
	End scanner.Location;  // location of closing "}" if present
}


func NewBlock(loc scanner.Location, tok int) *Block {
	assert(tok == token.LBRACE || tok == token.COLON);
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

	LabelDecl struct {
		Loc scanner.Location;  // location of ":"
		Label *Ident;
	};

	DeclarationStat struct {
		Decl Decl;
	};

	ExpressionStat struct {
		Loc scanner.Location;  // location of Tok
		Tok int;  // INC, DEC, RETURN, GO, DEFER
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
	
	ForStat struct {
		Loc scanner.Location;  // location of "for"
		Init Stat;
		Cond Expr;
		Post Stat;
		Body *Block;
	};

	CaseClause struct {
		Loc scanner.Location;  // position for "case" or "default"
		Expr Expr;  // nil means default case
		Body *Block;
	};

	SwitchStat struct {
		Loc scanner.Location;  // location of "switch"
		Init Stat;
		Tag Expr;
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
	
	EmptyStat struct {
		Loc scanner.Location;  // location of ";"
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
		Loc scanner.Location;
	};

	ImportDecl struct {
		Loc scanner.Location;  // if > 0: position of "import"
		Ident *Ident;
		Path Expr;
	};
	
	ConstDecl struct {
		Loc scanner.Location;  // if > 0: position of "const"
		Idents []*Ident;
		Typ Expr;
		Vals Expr;
		Comment CommentGroup;
	};
	
	TypeDecl struct {
		Loc scanner.Location;  // if > 0: position of "type"
		Ident *Ident;
		Typ Expr;
		Comment CommentGroup;
	};
	
	VarDecl struct {
		Loc scanner.Location;  // if > 0: position of "var"
		Idents []*Ident;
		Typ Expr;
		Vals Expr;
		Comment CommentGroup;
	};

	FuncDecl struct {
		Loc scanner.Location;  // location of "func"
		Recv *Field;
		Ident *Ident;
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
	Ident *Ident;
	Decls []Decl;
	Comment CommentGroup;
	Comments []CommentGroup;
}


func NewProgram(loc scanner.Location) *Program {
	p := new(Program);
	p.Loc = loc;
	return p;
}
