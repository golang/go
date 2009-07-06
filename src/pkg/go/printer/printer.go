// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The printer package implements printing of AST nodes.
package printer

import (
	"fmt";
	"go/ast";
	"go/token";
	"io";
	"os";
	"reflect";
	"strings";
)


// Printing is controlled with these flags supplied
// to Fprint via the mode parameter.
//
const (
	DocComments uint = 1 << iota;  // print documentation comments
	OptCommas;  // print optional commas
	OptSemis;  // print optional semicolons
)


type printer struct {
	// configuration (does not change after initialization)
	output io.Writer;
	mode uint;
	errors chan os.Error;
	comments ast.Comments;  // list of unassociated comments; or nil

	// current state (changes during printing)
	written int;  // number of bytes written
	level int;  // function nesting level; 0 = package scope, 1 = top-level function scope, etc.
	indent int;  // indent level
	pos token.Position;  // output position (possibly estimated) in "AST space"

	// comments
	cindex int;  // the current comment index
	cpos token.Position;  // the position of the next comment
}


func (p *printer) hasComment(pos token.Position) bool {
	return p.cpos.Offset < pos.Offset;
}


func (p *printer) nextComment() {
	p.cindex++;
	if p.comments != nil && p.cindex < len(p.comments) && p.comments[p.cindex] != nil {
		p.cpos = p.comments[p.cindex].Pos();
	} else {
		p.cpos = token.Position{1<<30, 1<<30, 1};  // infinite
	}
}


func (p *printer) setComments(comments ast.Comments) {
	p.comments = comments;
	p.cindex = -1;
	p.nextComment();
}


func (p *printer) init(output io.Writer, mode uint) {
	p.output = output;
	p.mode = mode;
	p.errors = make(chan os.Error);
	p.setComments(nil);
}


var (
	blank = []byte{' '};
	tab = []byte{'\t'};
	newline = []byte{'\n'};
	formfeed = []byte{'\f'};
)


// Writing to p.output is done with write0 which also handles errors.
// It should only be called by write.
//
func (p *printer) write0(data []byte) {
	n, err := p.output.Write(data);
	p.written += n;
	if err != nil {
		p.errors <- err;
	}
}


func (p *printer) write(data []byte) {
	i0 := 0;
	for i, b := range data {
		if b == '\n' || b == '\f' {
			// write segment ending in a newline/formfeed followed by indentation
			// TODO should convert '\f' into '\n' if the output is not going through
			//      tabwriter
			p.write0(data[i0 : i+1]);
			for j := p.indent; j > 0; j-- {
				p.write0(tab);
			}
			i0 = i+1;

			// update p.pos
			p.pos.Offset += i+1 - i0 + p.indent;
			p.pos.Line++;
			p.pos.Column = p.indent + 1;
		}
	}

	// write remaining segment
	p.write0(data[i0 : len(data)]);

	// update p.pos
	n := len(data) - i0;
	p.pos.Offset += n;
	p.pos.Column += n;
}


// Reduce contiguous sequences of '\t' in a []byte to a single '\t'.
func untabify(src []byte) []byte {
	dst := make([]byte, len(src));
	j := 0;
	for i, c := range src {
		if c != '\t' || i == 0 || src[i-1] != '\t' {
			dst[j] = c;
			j++;
		}
	}
	return dst[0 : j];
}


func (p *printer) adjustSpacingAndMergeComments() {
	for ; p.hasComment(p.pos); p.nextComment() {
		// we have a comment that comes before the current position
		comment := p.comments[p.cindex];
		p.write(untabify(comment.Text));
		// TODO
		// - classify comment and provide better formatting
		// - add extra newlines if so indicated by source positions
	}
}


func (p *printer) print(args ...) {
	v := reflect.NewValue(args).(reflect.StructValue);
	for i := 0; i < v.Len(); i++ {
		p.adjustSpacingAndMergeComments();
		f := v.Field(i);
		switch x := f.Interface().(type) {
		case int:
			// indentation delta
			p.indent += x;
			if p.indent < 0 {
				panic("print: negative indentation");
			}
		case []byte:
			p.write(x);
		case string:
			p.write(strings.Bytes(x));
		case token.Token:
			p.write(strings.Bytes(x.String()));
		case token.Position:
			// set current position
			p.pos = x;
		default:
			panicln("print: unsupported argument type", f.Type().String());
		}
	}
}


// ----------------------------------------------------------------------------
// Printing of common AST nodes.

func (p *printer) optSemis() bool {
	return p.mode & OptSemis != 0;
}


func (p *printer) comment(c *ast.Comment) {
	if c != nil {
		text := c.Text;
		if text[1] == '/' {
			// //-style comment - dont print the '\n'
			// TODO scanner should probably not include the '\n' in this case
			text = text[0 : len(text)-1];
		}
		p.print(tab, c.Pos(), text);  // tab-separated trailing comment
	}
}


func (p *printer) doc(d ast.Comments) {
	if p.mode & DocComments != 0 {
		for _, c := range d {
			p.print(c.Pos(), c.Text);
		}
	}
}


func (p *printer) expr(x ast.Expr) bool

func (p *printer) identList(list []*ast.Ident) {
	for i, x := range list {
		if i > 0 {
			p.print(token.COMMA, blank);
		}
		p.expr(x);
	}
}


func (p *printer) exprList(list []ast.Expr) {
	for i, x := range list {
		if i > 0 {
			p.print(token.COMMA, blank);
		}
		p.expr(x);
	}
}


func (p *printer) parameters(list []*ast.Field) {
	p.print(token.LPAREN);
	if len(list) > 0 {
		p.level++;  // adjust nesting level for parameters
		for i, par := range list {
			if i > 0 {
				p.print(token.COMMA, blank);
			}
			p.identList(par.Names);  // p.level > 0; all identifiers will be printed
			if len(par.Names) > 0 {
				// at least one identifier
				p.print(blank);
			};
			p.expr(par.Type);
		}
		p.level--;
	}
	p.print(token.RPAREN);
}


func (p *printer) signature(params, result []*ast.Field) {
	p.parameters(params);
	if result != nil {
		p.print(blank);

		if len(result) == 1 && result[0].Names == nil {
			// single anonymous result; no ()'s unless it's a function type
			f := result[0];
			if _, isFtyp := f.Type.(*ast.FuncType); !isFtyp {
				p.expr(f.Type);
				return;
			}
		}

		p.parameters(result);
	}
}


// Returns true if the field list ends in a closing brace.
func (p *printer) fieldList(lbrace token.Position, list []*ast.Field, rbrace token.Position, isInterface bool) bool {
	if !lbrace.IsValid() {
		// forward declaration without {}'s
		return false;  // no {}'s
	}

	if len(list) == 0 {
		p.print(blank, lbrace, token.LBRACE, rbrace, token.RBRACE);
		return true;  // empty list with {}'s
	}

	p.print(blank, lbrace, token.LBRACE, +1, newline);

	var lastWasAnon bool;  // true if the previous line was an anonymous field
	var lastComment *ast.Comment;  // the comment from the previous line
	for i, f := range list {
		// at least one visible identifier or anonymous field
		isAnon := len(f.Names) == 0;
		if i > 0 {
			p.print(token.SEMICOLON);
			p.comment(lastComment);
			if lastWasAnon == isAnon {
				// previous and current line have same structure;
				// continue with existing columns
				p.print(newline);
			} else {
				// previous and current line have different structure;
				// flush tabwriter and start new columns (the "type
				// column" on a line with named fields may line up
				// with the "trailing comment column" on a line with
				// an anonymous field, leading to bad alignment)
				p.print(formfeed);
			}
		}

		p.doc(f.Doc);
		if !isAnon {
			p.identList(f.Names);
			p.print(tab);
		}

		if isInterface {
			if ftyp, isFtyp := f.Type.(*ast.FuncType); isFtyp {
				// methods
				p.signature(ftyp.Params, ftyp.Results);
			} else {
				// embedded interface
				p.expr(f.Type);
			}
		} else {
			p.expr(f.Type);
			if f.Tag != nil {
				p.print(tab);
				p.expr(&ast.StringList{f.Tag});
			}
		}

		lastWasAnon = isAnon;
		lastComment = f.Comment;
	}

	if p.optSemis() {
		p.print(token.SEMICOLON);
	}
	p.comment(lastComment);

	p.print(-1, newline, rbrace, token.RBRACE);

	return true;  // field list with {}'s
}


// ----------------------------------------------------------------------------
// Expressions

func (p *printer) stmt(s ast.Stmt) (optSemi bool)

// Returns true if a separating semicolon is optional.
func (p *printer) expr1(expr ast.Expr, prec1 int) (optSemi bool) {
	p.print(expr.Pos());

	switch x := expr.(type) {
	case *ast.BadExpr:
		p.print("BadExpr");

	case *ast.Ident:
		p.print(x.Value);

	case *ast.BinaryExpr:
		prec := x.Op.Precedence();
		if prec < prec1 {
			p.print(token.LPAREN);
		}
		p.expr1(x.X, prec);
		p.print(blank, x.OpPos, x.Op, blank);
		p.expr1(x.Y, prec);
		if prec < prec1 {
			p.print(token.RPAREN);
		}

	case *ast.KeyValueExpr:
		p.expr(x.Key);
		p.print(blank, x.Colon, token.COLON, blank);
		p.expr(x.Value);

	case *ast.StarExpr:
		p.print(token.MUL);
		p.expr(x.X);

	case *ast.UnaryExpr:
		prec := token.UnaryPrec;
		if prec < prec1 {
			p.print(token.LPAREN);
		}
		p.print(x.Op);
		if x.Op == token.RANGE {
			p.print(blank);
		}
		p.expr1(x.X, prec);
		if prec < prec1 {
			p.print(token.RPAREN);
		}

	case *ast.IntLit:
		p.print(x.Value);

	case *ast.FloatLit:
		p.print(x.Value);

	case *ast.CharLit:
		p.print(x.Value);

	case *ast.StringLit:
		p.print(x.Value);

	case *ast.StringList:
		for i, x := range x.Strings {
			if i > 0 {
				p.print(blank);
			}
			p.expr(x);
		}

	case *ast.FuncLit:
		p.expr(x.Type);
		p.print(blank);
		p.level++;  // adjust nesting level for function body
		p.stmt(x.Body);
		p.level--;

	case *ast.ParenExpr:
		p.print(token.LPAREN);
		p.expr(x.X);
		p.print(x.Rparen, token.RPAREN);

	case *ast.SelectorExpr:
		p.expr1(x.X, token.HighestPrec);
		p.print(token.PERIOD);
		p.expr1(x.Sel, token.HighestPrec);

	case *ast.TypeAssertExpr:
		p.expr1(x.X, token.HighestPrec);
		p.print(token.PERIOD, token.LPAREN);
		p.expr(x.Type);
		p.print(token.RPAREN);

	case *ast.IndexExpr:
		p.expr1(x.X, token.HighestPrec);
		p.print(token.LBRACK);
		p.expr(x.Index);
		if x.End != nil {
			p.print(blank, token.COLON, blank);
			p.expr(x.End);
		}
		p.print(token.RBRACK);

	case *ast.CallExpr:
		p.expr1(x.Fun, token.HighestPrec);
		p.print(x.Lparen, token.LPAREN);
		p.exprList(x.Args);
		p.print(x.Rparen, token.RPAREN);

	case *ast.CompositeLit:
		p.expr1(x.Type, token.HighestPrec);
		p.print(x.Lbrace, token.LBRACE);
		p.exprList(x.Elts);
		if p.mode & OptCommas != 0 {
			p.print(token.COMMA);
		}
		p.print(x.Rbrace, token.RBRACE);

	case *ast.Ellipsis:
		p.print(token.ELLIPSIS);

	case *ast.ArrayType:
		p.print(token.LBRACK);
		if x.Len != nil {
			p.expr(x.Len);
		}
		p.print(token.RBRACK);
		p.expr(x.Elt);

	case *ast.StructType:
		p.print(token.STRUCT);
		optSemi = p.fieldList(x.Lbrace, x.Fields, x.Rbrace, false);

	case *ast.FuncType:
		p.print(token.FUNC);
		p.signature(x.Params, x.Results);

	case *ast.InterfaceType:
		p.print(token.INTERFACE);
		optSemi = p.fieldList(x.Lbrace, x.Methods, x.Rbrace, true);

	case *ast.MapType:
		p.print(token.MAP, blank, token.LBRACK);
		p.expr(x.Key);
		p.print(token.RBRACK);
		p.expr(x.Value);

	case *ast.ChanType:
		switch x.Dir {
		case ast.SEND | ast.RECV:
			p.print(token.CHAN);
		case ast.RECV:
			p.print(token.ARROW, token.CHAN);
		case ast.SEND:
			p.print(token.CHAN, blank, token.ARROW);
		}
		p.print(blank);
		p.expr(x.Value);

	default:
		panic("unreachable");
	}

	return optSemi;
}


// Returns true if a separating semicolon is optional.
func (p *printer) expr(x ast.Expr) bool {
	return p.expr1(x, token.LowestPrec);
}


// ----------------------------------------------------------------------------
// Statements

func (p *printer) decl(decl ast.Decl) (optSemi bool)

// Print the statement list indented, but without a newline after the last statement.
func (p *printer) stmtList(list []ast.Stmt) {
	if len(list) > 0 {
		p.print(+1, newline);
		optSemi := false;
		for i, s := range list {
			if i > 0 {
				if !optSemi || p.optSemis() {
					// semicolon is required
					p.print(token.SEMICOLON);
				}
				p.print(newline);
			}
			optSemi = p.stmt(s);
		}
		if p.optSemis() {
			p.print(token.SEMICOLON);
		}
		p.print(-1);
	}
}


func (p *printer) block(s *ast.BlockStmt) {
	p.print(s.Pos(), token.LBRACE);
	if len(s.List) > 0 {
		p.stmtList(s.List);
		p.print(newline);
	}
	p.print(s.Rbrace, token.RBRACE);
}


func (p *printer) switchBlock(s *ast.BlockStmt) {
	p.print(s.Pos(), token.LBRACE);
	if len(s.List) > 0 {
		for i, s := range s.List {
			// s is one of *ast.CaseClause, *ast.TypeCaseClause, *ast.CommClause;
			p.print(newline);
			p.stmt(s);
		}
		p.print(newline);
	}
	p.print(s.Rbrace, token.RBRACE);
}


func (p *printer) controlClause(isForStmt bool, init ast.Stmt, expr ast.Expr, post ast.Stmt) {
	if init == nil && post == nil {
		// no semicolons required
		if expr != nil {
			p.print(blank);
			p.expr(expr);
		}
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		p.print(blank);
		if init != nil {
			p.stmt(init);
		}
		p.print(token.SEMICOLON, blank);
		if expr != nil {
			p.expr(expr);
		}
		if isForStmt {
			p.print(token.SEMICOLON, blank);
			if post != nil {
				p.stmt(post);
			}
		}
	}
}


// Returns true if a separating semicolon is optional.
func (p *printer) stmt(stmt ast.Stmt) (optSemi bool) {
	p.print(stmt.Pos());

	switch s := stmt.(type) {
	case *ast.BadStmt:
		p.print("BadStmt");

	case *ast.DeclStmt:
		optSemi = p.decl(s.Decl);

	case *ast.EmptyStmt:
		// nothing to do

	case *ast.LabeledStmt:
		p.print(-1, newline);
		p.expr(s.Label);
		p.print(token.COLON, tab, +1);
		optSemi = p.stmt(s.Stmt);

	case *ast.ExprStmt:
		p.expr(s.X);

	case *ast.IncDecStmt:
		p.expr(s.X);
		p.print(s.Tok);

	case *ast.AssignStmt:
		p.exprList(s.Lhs);
		p.print(blank, s.TokPos, s.Tok, blank);
		p.exprList(s.Rhs);

	case *ast.GoStmt:
		p.print(token.GO, blank);
		p.expr(s.Call);

	case *ast.DeferStmt:
		p.print(token.DEFER, blank);
		p.expr(s.Call);

	case *ast.ReturnStmt:
		p.print(token.RETURN);
		if s.Results != nil {
			p.print(blank);
			p.exprList(s.Results);
		}

	case *ast.BranchStmt:
		p.print(s.Tok);
		if s.Label != nil {
			p.print(blank);
			p.expr(s.Label);
		}

	case *ast.BlockStmt:
		p.block(s);
		optSemi = true;

	case *ast.IfStmt:
		p.print(token.IF);
		p.controlClause(false, s.Init, s.Cond, nil);
		p.print(blank);
		p.block(s.Body);
		optSemi = true;
		if s.Else != nil {
			p.print(blank, token.ELSE, blank);
			optSemi = p.stmt(s.Else);
		}

	case *ast.CaseClause:
		if s.Values != nil {
			p.print(token.CASE, blank);
			p.exprList(s.Values);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body);

	case *ast.SwitchStmt:
		p.print(token.SWITCH);
		p.controlClause(false, s.Init, s.Tag, nil);
		p.print(blank);
		p.switchBlock(s.Body);
		optSemi = true;

	case *ast.TypeCaseClause:
		if s.Type != nil {
			p.print(token.CASE, blank);
			p.expr(s.Type);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body);

	case *ast.TypeSwitchStmt:
		p.print(token.SWITCH);
		if s.Init != nil {
			p.print(blank);
			p.stmt(s.Init);
			p.print(token.SEMICOLON);
		}
		p.print(blank);
		p.stmt(s.Assign);
		p.print(blank);
		p.switchBlock(s.Body);
		optSemi = true;

	case *ast.CommClause:
		if s.Rhs != nil {
			p.print(token.CASE, blank);
			if s.Lhs != nil {
				p.expr(s.Lhs);
				p.print(blank, s.Tok, blank);
			}
			p.expr(s.Rhs);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body);

	case *ast.SelectStmt:
		p.print(token.SELECT, blank);
		p.switchBlock(s.Body);
		optSemi = true;

	case *ast.ForStmt:
		p.print(token.FOR);
		p.controlClause(true, s.Init, s.Cond, s.Post);
		p.print(blank);
		p.block(s.Body);
		optSemi = true;

	case *ast.RangeStmt:
		p.print(token.FOR, blank);
		p.expr(s.Key);
		if s.Value != nil {
			p.print(token.COMMA, blank);
			p.expr(s.Value);
		}
		p.print(blank, s.TokPos, s.Tok, blank, token.RANGE, blank);
		p.expr(s.X);
		p.print(blank);
		p.block(s.Body);
		optSemi = true;

	default:
		panic("unreachable");
	}

	return optSemi;
}


// ----------------------------------------------------------------------------
// Declarations

// Returns true if a separating semicolon is optional.
func (p *printer) spec(spec ast.Spec) (optSemi bool) {
	switch s := spec.(type) {
	case *ast.ImportSpec:
		p.doc(s.Doc);
		if s.Name != nil {
			p.expr(s.Name);
		}
		// TODO fix for longer package names
		p.print(tab, s.Path[0].Pos(), s.Path[0].Value);

	case *ast.ValueSpec:
		p.doc(s.Doc);
		p.identList(s.Names);
		if s.Type != nil {
			p.print(blank);  // TODO switch to tab? (indent problem with structs)
			p.expr(s.Type);
		}
		if s.Values != nil {
			p.print(tab, token.ASSIGN, blank);
			p.exprList(s.Values);
		}

	case *ast.TypeSpec:
		p.doc(s.Doc);
		p.expr(s.Name);
		p.print(blank);  // TODO switch to tab? (but indent problem with structs)
		optSemi = p.expr(s.Type);

	default:
		panic("unreachable");
	}

	return optSemi;
}


// Returns true if a separating semicolon is optional.
func (p *printer) decl(decl ast.Decl) (optSemi bool) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		p.print(d.Pos(), "BadDecl");

	case *ast.GenDecl:
		p.doc(d.Doc);
		p.print(d.Pos(), d.Tok, blank);

		if d.Lparen.IsValid() {
			// group of parenthesized declarations
			p.print(d.Lparen, token.LPAREN, +1, newline);
			for i, s := range d.Specs {
				if i > 0 {
					p.print(token.SEMICOLON, newline);
				}
				p.spec(s);
			}
			if p.optSemis() {
				p.print(token.SEMICOLON);
			}
			p.print(-1, newline, d.Rparen, token.RPAREN);
			optSemi = true;

		} else {
			// single declaration
			optSemi = p.spec(d.Specs[0]);
		}

	case *ast.FuncDecl:
		p.doc(d.Doc);
		p.print(d.Pos(), token.FUNC, blank);
		if recv := d.Recv; recv != nil {
			// method: print receiver
			p.print(token.LPAREN);
			if len(recv.Names) > 0 {
				p.expr(recv.Names[0]);
				p.print(blank);
			}
			p.expr(recv.Type);
			p.print(token.RPAREN, blank);
		}
		p.expr(d.Name);
		p.signature(d.Type.Params, d.Type.Results);
		if d.Body != nil {
			p.print(blank);
			p.level++;  // adjust nesting level for function body
			p.stmt(d.Body);
			p.level--;
		}

	default:
		panic("unreachable");
	}

	return optSemi;
}


// ----------------------------------------------------------------------------
// Programs

func (p *printer) program(prog *ast.Program) {
	// set unassociated comments
	// TODO enable this once comments are properly interspersed
	// p.setComments(prog.Comments);

	p.doc(prog.Doc);
	p.print(prog.Pos(), token.PACKAGE, blank);
	p.expr(prog.Name);

	for _, d := range prog.Decls {
		p.print(newline, newline);
		p.decl(d);
		if p.optSemis() {
			p.print(token.SEMICOLON);
		}
	}

	p.print(newline);
}


// ----------------------------------------------------------------------------
// Public interface

// Fprint "pretty-prints" an AST node to output and returns the number of
// bytes written, and an error, if any. The node type must be *ast.Program,
// or assignment-compatible to ast.Expr, ast.Decl, or ast.Stmt. Printing is
// controlled by the mode parameter. For best results, the output should be
// a tabwriter.Writer.
//
func Fprint(output io.Writer, node interface{}, mode uint) (int, os.Error) {
	var p printer;
	p.init(output, mode);

	go func() {
		switch n := node.(type) {
		case ast.Expr:
			p.expr(n);
		case ast.Stmt:
			p.stmt(n);
		case ast.Decl:
			p.decl(n);
		case *ast.Program:
			p.program(n);
		default:
			p.errors <- os.NewError("unsupported node type");
		}
		p.errors <- nil;  // no errors
	}();
	err := <-p.errors;  // wait for completion of goroutine

	return p.written, err;
}
