// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format

import (
	"fmt";
	"go/scanner";
	"go/token";
	"io";
	"reflect";
	"os";
)


// -----------------------------------------------------------------------------
// Format

// node kind
const (
	self = iota;
	alternative;
	sequence;
	field;
	literal;
	option;
	repetition;
)


type node struct {
	kind int;
	name string;  // field name
	value []byte;  // literal value
	x, y *node;
}


// A Format is a set of production nodes.
type Format map [string] *node;


// -----------------------------------------------------------------------------
// Parsing

/*	Format      = { Production } .
	Production  = DottedName [ "=" Expression ] "." .
	DottedName  = name { "." name } .
	Expression  = Term { "|" Term } .
	Term        = Factor { Factor } .
	Factor      = "*" | name | string_literal | Group | Option | Repetition .
	Group       = "(" Expression ")" .
	Option      = "[" Expression "]" .
	Repetition  = "{" Expression "}" .
*/


type parser struct {
	scanner scanner.Scanner;
	
	// error handling
	lastline int;  // > 0 if there was any error

	// next token
	pos token.Position;  // token position
	tok token.Token;  // one token look-ahead
	lit []byte;  // token literal
}


// The parser implements the scanner.ErrorHandler interface.
func (p *parser) Error(pos token.Position, msg string) {
	if pos.Line != p.lastline {
		// only report error if not on the same line as previous error
		// in the hope to reduce number of follow-up errors reported
		fmt.Fprintf(os.Stderr, "%d:%d: %s\n", pos.Line, pos.Column, msg);
	}
	p.lastline = pos.Line;
}


func (p *parser) next() {
	p.pos, p.tok, p.lit = p.scanner.Scan();
}


func (p *parser) error_expected(pos token.Position, msg string) {
	msg = "expected " + msg;
	if pos.Offset == p.pos.Offset {
		// the error happened at the current position;
		// make the error message more specific
		msg += ", found '" + p.tok.String() + "'";
		if p.tok.IsLiteral() {
			msg += " " + string(p.lit);
		}
	}
	p.Error(pos, msg);
}


func (p *parser) expect(tok token.Token) token.Position {
	pos := p.pos;
	if p.tok != tok {
		p.error_expected(pos, "'" + tok.String() + "'");
	}
	p.next();  // make progress in any case
	return pos;
}


func (p *parser) parseName() string {
	name := string(p.lit);
	p.expect(token.IDENT);
	return name;
}


func (p *parser) parseDottedName() string {
	name := p.parseName();
	for p.tok == token.PERIOD {
		p.next();
		name = name + "." + p.parseName();
	}
	return name;
}


// TODO should have WriteByte in ByteBuffer instead!
var (
	newlineByte = []byte{'\n'};
	tabByte = []byte{'\t'};
)


func escapeString(s []byte) []byte {
	// the string syntax is correct since it comes from the scannner
	var buf io.ByteBuffer;
	i0 := 0;
	for i := 0; i < len(s); {
		if s[i] == '\\' {
			buf.Write(s[i0 : i]);
			i++;
			switch s[i] {
			case 'n':
				buf.Write(newlineByte);
			case 't':
				buf.Write(tabByte);
			default:
				panic("unhandled escape:", string(s[i]));
			}
			i++;
			i0 = i;
		} else {
			i++;
		}
	}
	
	if i0 == 0 {
		// no escape sequences
		return s;
	}

	buf.Write(s[i0 : len(s)]);
	return buf.Data();
}


func (p *parser) parseValue() []byte {
	if p.tok != token.STRING {
		p.expect(token.STRING);
		return nil;
	}

	s := p.lit[1 : len(p.lit)-1];  // strip quotes
	if p.lit[0] == '"' {
		s = escapeString(s);
	}

	p.next();
	return s;
}


func (p *parser) parseExpression() *node

func (p *parser) parseFactor() (x *node) {
	switch p.tok {
	case token.MUL:
		x = &node{self, "", nil, nil, nil};

	case token.IDENT:
		x = &node{field, p.parseName(), nil, nil, nil};

	case token.STRING:
		x = &node{literal, "", p.parseValue(), nil, nil};

	case token.LPAREN:
		p.next();
		x = p.parseExpression();
		p.expect(token.RPAREN);

	case token.LBRACK:
		p.next();
		x = &node{option, "", nil, p.parseExpression(), nil};
		p.expect(token.RBRACK);

	case token.LBRACE:
		p.next();
		x = &node{repetition, "", nil, p.parseExpression(), nil};
		p.expect(token.RBRACE);

	default:
		p.error_expected(p.pos, "factor");
		p.next();  // make progress
	}

	return x;
}


func (p *parser) parseTerm() *node {
	x := p.parseFactor();

	for	p.tok == token.IDENT ||
		p.tok == token.STRING ||
		p.tok == token.LPAREN ||
		p.tok == token.LBRACK ||
		p.tok == token.LBRACE
	{
		y := p.parseFactor();
		x = &node{sequence, "", nil, x, y};
	}

	return x;
}


func (p *parser) parseExpression() *node {
	x := p.parseTerm();

	for p.tok == token.OR {
		p.next();
		y := p.parseTerm();
		x = &node{alternative, "", nil, x, y};
	}

	return x;
}


func (p *parser) parseProduction() (string, *node) {
	name := p.parseDottedName();
	
	var x *node;
	if p.tok == token.ASSIGN {
		p.next();
		x = p.parseExpression();
	}

	p.expect(token.PERIOD);

	return name, x;
}


func (p *parser) parseFormat() Format {
	format := make(Format);

	prefix := "";
	for p.tok != token.EOF {
		pos := p.pos;
		name, x := p.parseProduction();
		if x == nil {
			// prefix declaration
			prefix = name + ".";
		} else {
			// production declaration
			// add package prefix, if any
			if prefix != "" {
				name = prefix + name;
			}
			// add production to format
			if t, found := format[name]; !found {
				format[name] = x;
			} else {
				p.Error(pos, "production already declared: " + name);
			}
		}
	}
	p.expect(token.EOF);

	return format;
}


func readSource(src interface{}, err scanner.ErrorHandler) []byte {
	errmsg := "invalid input type (or nil)";

	switch s := src.(type) {
	case string:
		return io.StringBytes(s);
	case []byte:
		return s;
	case *io.ByteBuffer:
		// is io.Read, but src is already available in []byte form
		if s != nil {
			return s.Data();
		}
	case io.Read:
		var buf io.ByteBuffer;
		n, os_err := io.Copy(s, &buf);
		if os_err == nil {
			return buf.Data();
		}
		errmsg = os_err.String();
	}

	if err != nil {
		// TODO fix this
		panic();
		//err.Error(noPos, errmsg);
	}
	return nil;
}


func Parse(src interface{}) Format {
	// initialize parser
	var p parser;
	p.scanner.Init(readSource(src, &p), &p, false);
	p.next();

	f := p.parseFormat();

	if p.lastline > 0 {	
		return nil;  // src contains errors
	}
	return f;
}


// -----------------------------------------------------------------------------
// Application

func fieldIndex(v reflect.StructValue, fieldname string) int {
	t := v.Type().(reflect.StructType);
	for i := 0; i < v.Len(); i++ {
		name, typ, tag, offset := t.Field(i);
		if name == fieldname {
			return i;
		}
	}
	return -1;
}


func getField(v reflect.StructValue, fieldname string) reflect.Value {
	i := fieldIndex(v, fieldname);
	if i < 0 {
		panicln("field not found:", fieldname);
	}

	return v.Field(i);
}


func (f Format) apply(w io.Write, v reflect.Value) bool

// Returns true if a non-empty field value was found.
func (f Format) print(w io.Write, x *node, v reflect.Value, index int) bool {
	switch x.kind {
	case self:
		panic("self");

	case alternative:
		// print the contents of the first alternative with a non-empty field
		var buf io.ByteBuffer;
		if !f.print(&buf, x.x, v, -1) {
			f.print(&buf, x.y, v, -1);
		}
		w.Write(buf.Data());

	case sequence:
		f.print(w, x.x, v, -1);
		f.print(w, x.y, v, -1);

	case field:
		if sv, is_struct := v.(reflect.StructValue); is_struct {
			return f.apply(w, getField(sv, x.name));
		} else {
			panicln("not in a struct - field:", x.name);
		}

	case literal:
		w.Write(x.value);

	case option:
		// print the contents of the option if there is a non-empty field
		var buf io.ByteBuffer;
		if f.print(&buf, x.x, v, -1) {
			w.Write(buf.Data());
		}

	case repetition:
		// print the contents of the repetition while there is a non-empty field
		for i := 0; ; i++ {
			var buf io.ByteBuffer;
			if f.print(&buf, x.x, v, i) {
				w.Write(buf.Data());
			} else {
				break;
			}
		}

	default:
		panic("unreachable");
	}

	return false;
}


func (f Format) Dump() {
	for name, x := range f {
		println(name, x);
	}
}


func (f Format) apply(w io.Write, v reflect.Value) bool {
	println("apply typename:", v.Type().Name());

	if x, found := f[v.Type().Name()]; found {
		// format using corresponding production
		f.print(w, x, v, -1);
		
	} else {
		// format using default formats
		switch x := v.(type) {
		case reflect.ArrayValue:
			if x.Len() == 0 {
				return false;
			}
			for i := 0; i < x.Len(); i++ {
				f.apply(w, x.Elem(i));
			}

		case reflect.StringValue:
			w.Write(io.StringBytes(x.Get()));

		case reflect.IntValue:
			// TODO is this the correct way to check the right type?
			// or should it be t, ok := x.Interface().(token.Token) instead?
			if x.Type().Name() == "token.Token" {
				fmt.Fprintf(w, "%s", token.Token(x.Get()).String());
			} else {
				fmt.Fprintf(w, "%d", x.Get());
			}

		case reflect.InterfaceValue:
			f.apply(w, x.Value());

		case reflect.PtrValue:
			// TODO is this the correct way to check nil ptr?
			if x.Get() == nil {
				return false;
			}
			return f.apply(w, x.Sub());

		default:
			panicln("unsupported kind:", v.Kind());
		}
	}

	return true;
}


func (f Format) Apply(w io.Write, data interface{}) {
	f.apply(w, reflect.NewValue(data));
}
