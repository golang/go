// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import (
	"vector";
	Scanner "scanner";
	SymbolTable "symboltable";
)


type (
	Block struct;
	Expr interface;
	Decl struct;
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
// Types

const /* form */ (
	// BADTYPE types are compatible with any type and don't cause further errors.
	// They are introduced only as a result of an error in the source code. A
	// correct program cannot have BAD types.
	BADTYPE = iota;

	// A type name
	TYPENAME;

	// composite types
	ARRAY; STRUCT; INTERFACE; MAP; CHANNEL; FUNCTION; POINTER;

	// open-ended parameter type
	ELLIPSIS
)


func FormStr(form int) string {
	switch form {
	case BADTYPE: return "BADTYPE";
	case TYPENAME: return "TYPENAME";
	case ARRAY: return "ARRAY";
	case STRUCT: return "STRUCT";
	case INTERFACE: return "INTERFACE";
	case MAP: return "MAP";
	case CHANNEL: return "CHANNEL";
	case FUNCTION: return "FUNCTION";
	case POINTER: return "POINTER";
	case ELLIPSIS: return "ELLIPSIS";
	}
	return "<unknown Type form>";
}


const /* channel mode */ (
	FULL = iota;
	SEND;
	RECV;
)


type Type struct {
	Id int;  // unique id

	Form int;  // type form
	Size int;  // size in bytes
	Scope *SymbolTable.Scope;  // locals, fields & methods

	// syntactic components
	Pos int;  // source position (< 0 if unknown position)
	Expr Expr;  // type name, vector length
	Mode int;  // channel mode
	Key *Type;  // receiver type or map key
	Elt *Type;  // type name type, vector, map, channel or pointer element type, function result type
	List *vector.Vector; End int;  // struct fields, interface methods, function parameters
}


var typeId int;

func NewType(pos, form int) *Type {
	typ := new(Type);
	typ.Id = typeId;
	typeId++;

	typ.Pos = pos;
	typ.Form = form;

	return typ;
}


func (typ* Type) String() string {
	if typ != nil {
		return
			"Type(" +
			FormStr(typ.Form) +
			")";
	}
	return "nil";
}


var BadType = NewType(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Expressions

type (
	ExprVisitor interface;

	Expr interface {
		Pos() int;
		Visit(v ExprVisitor);
	};

	BadExpr struct {
		Pos_ int;
	};

	Ident struct {
		Pos_ int;
		Obj *SymbolTable.Object;
	};

	BinaryExpr struct {
		Pos_, Tok int;
		X, Y Expr;
	};

	UnaryExpr struct {
		Pos_, Tok int;
		X Expr;
	};

	BasicLit struct {
		Pos_, Tok int;
		Val string
	};

	FunctionLit struct {
		Pos_ int;  // position of "func"
		Typ *Type;
		Body *Block;
	};
	
	TypeLit struct {
		Typ *Type;
	};

	Selector struct {
		Pos_ int;  // position of "."
		X Expr;
		Sel *Ident;
	};

	TypeGuard struct {
		Pos_ int;  // position of "."
		X Expr;
		Typ *Type;
	};

	Index struct {
		Pos_ int;  // position of "["
		X, I Expr;
	};
	
	Call struct {
		Pos_ int;  // position of "("
		F, Args Expr
	};
)


type ExprVisitor interface {
	DoBadExpr(x *BadExpr);
	DoIdent(x *Ident);
	DoBinaryExpr(x *BinaryExpr);
	DoUnaryExpr(x *UnaryExpr);
	DoBasicLit(x *BasicLit);
	DoFunctionLit(x *FunctionLit);
	DoTypeLit(x *TypeLit);
	DoSelector(x *Selector);
	DoTypeGuard(x *TypeGuard);
	DoIndex(x *Index);
	DoCall(x *Call);
}


func (x *BadExpr) Pos() int { return x.Pos_; }
func (x *Ident) Pos() int { return x.Pos_; }
func (x *BinaryExpr) Pos() int { return x.Pos_; }
func (x *UnaryExpr) Pos() int { return x.Pos_; }
func (x *BasicLit) Pos() int { return x.Pos_; }
func (x *FunctionLit) Pos() int { return x.Pos_; }
func (x *TypeLit) Pos() int { return x.Typ.Pos; }
func (x *Selector) Pos() int { return x.Pos_; }
func (x *TypeGuard) Pos() int { return x.Pos_; }
func (x *Index) Pos() int { return x.Pos_; }
func (x *Call) Pos() int { return x.Pos_; }


func (x *BadExpr) Visit(v ExprVisitor) { v.DoBadExpr(x); }
func (x *Ident) Visit(v ExprVisitor) { v.DoIdent(x); }
func (x *BinaryExpr) Visit(v ExprVisitor) { v.DoBinaryExpr(x); }
func (x *UnaryExpr) Visit(v ExprVisitor) { v.DoUnaryExpr(x); }
func (x *BasicLit) Visit(v ExprVisitor) { v.DoBasicLit(x); }
func (x *FunctionLit) Visit(v ExprVisitor) { v.DoFunctionLit(x); }
func (x *TypeLit) Visit(v ExprVisitor) { v.DoTypeLit(x); }
func (x *Selector) Visit(v ExprVisitor) { v.DoSelector(x); }
func (x *TypeGuard) Visit(v ExprVisitor) { v.DoTypeGuard(x); }
func (x *Index) Visit(v ExprVisitor) { v.DoIndex(x); }
func (x *Call) Visit(v ExprVisitor) { v.DoCall(x); }



// Length of a comma-separated expression list.
func ExprLen(x Expr) int {
	if x == nil {
		return 0;
	}
	n := 1;
	for {
		if p, ok := x.(*BinaryExpr); ok && p.Tok == Scanner.COMMA {
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
		assert(x.(*BinaryExpr).Tok == Scanner.COMMA);
		x = x.(*BinaryExpr).Y;
	}
	if t, is_binary := x.(*BinaryExpr); is_binary && t.Tok == Scanner.COMMA {
		x = t.X;
	}
	return x;
}


func (t *Type) Nfields() int {
	if t.List == nil {
		return 0;
	}
	nx, nt := 0, 0;
	for i, n := 0, t.List.Len(); i < n; i++ {
		if dummy, ok := t.List.At(i).(*TypeLit); ok {
			nt++;
		} else {
			nx++;
		}
	}
	if nx == 0 {
		return nt;
	}
	return nx;
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
	assert(tok == Scanner.LBRACE || tok == Scanner.COLON);
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
		Decl *Decl;
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

type Decl struct {
	Node;
	Ident Expr;  // nil for ()-style declarations
	Typ *Type;
	Val Expr;
	Body *Block;
	// list of *Decl for ()-style declarations
	List *vector.Vector; End int;
}


func NewDecl(pos, tok int) *Decl {
	d := new(Decl);
	d.Pos, d.Tok = pos, tok;
	return d;
}


var BadDecl = NewDecl(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Program

type Comment struct {
	Pos int;
	Text string;
}


func NewComment(pos int, text string) *Comment {
	c := new(Comment);
	c.Pos, c.Text = pos, text;
	return c;
}


type Program struct {
	Pos int;  // tok is Scanner.PACKAGE
	Ident Expr;
	Decls *vector.Vector;
	Comments *vector.Vector;
}


func NewProgram(pos int) *Program {
	p := new(Program);
	p.Pos = pos;
	return p;
}
