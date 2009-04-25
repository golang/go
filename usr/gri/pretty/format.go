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

// A production expression is built from the following nodes.
//
type (
	expr interface {
		implements_expr();
	};

	empty struct {
	};

	alternative struct {
		x, y expr;
	};

	sequence struct {
		x, y expr;
	};

	field struct {
		name string;  // including "^", "*"
		format expr;  // nil if no format specified
	};
	
	literal struct {
		// TODO should there be other types or should it all be string literals?
		value []byte;
	};

	option struct {
		x expr
	};

	repetition struct {
		x expr
	};

	// TODO custom formats are not yet used
	custom struct {
		name string;
		f func(w io.Write, value interface{}, name string) bool
	};
)


// These methods are used to enforce the "implements" relationship for
// better compile-time type checking.
//
// TODO If we had a basic accessor mechanism in the language (a field
// "f T" automatically implements a corresponding accessor "f() T", this
// could be expressed more easily by simply providing the field.
//
func (x *empty) implements_expr()  {}
func (x *alternative) implements_expr()  {}
func (x *sequence) implements_expr()  {}
func (x *field) implements_expr()  {}
func (x *literal) implements_expr()  {}
func (x *option) implements_expr()  {}
func (x *repetition) implements_expr()  {}
func (x *custom) implements_expr()  {}


// A Format is a set of production expressions.
type Format map [string] expr;


// -----------------------------------------------------------------------------
// Parsing

/*	TODO
	- EBNF vs Kleene notation
	- default formatters for basic types (may imply scopes so we can override)
	- installable custom formatters (like for template.go)
	- format strings
*/

/*	Format      = { Production } .
	Production  = Name [ "=" [ Expression ] ] ";" .
	Name        = identifier { "." identifier } .
	Expression  = Term { "|" Term } .
	Term        = Factor { Factor } .
	Factor      = string_literal | Field | Group | Option | Repetition .
	Field		= ( "^" | "*" | Name ) [ ":" Expression ] .
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


func (p *parser) parseIdentifier() string {
	name := string(p.lit);
	p.expect(token.IDENT);
	return name;
}


func (p *parser) parseName() string {
	name := p.parseIdentifier();
	for p.tok == token.PERIOD {
		p.next();
		name = name + "." + p.parseIdentifier();
	}
	return name;
}


// TODO WriteByte should be a ByteBuffer method
func writeByte(buf *io.ByteBuffer, b byte) {
	buf.Write([]byte{b});
}


// TODO make this complete
func escapeString(s []byte) []byte {
	// the string syntax is correct since it comes from the scannner
	var buf io.ByteBuffer;
	i0 := 0;
	for i := 0; i < len(s); {
		if s[i] == '\\' {
			buf.Write(s[i0 : i]);
			i++;
			var esc byte;
			switch s[i] {
			case 'n': esc = '\n';
			case 't': esc = '\t';
			default: panic("unhandled escape:", string(s[i]));
			}
			writeByte(&buf, esc);
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


func (p *parser) parseExpr() expr

func (p *parser) parseField() expr {
	var name string;
	switch p.tok {
	case token.XOR:
		name = "^";
		p.next();
	case token.MUL:
		name = "*";
		p.next();
	case token.IDENT:
		name = p.parseName();
	default:
		panic("unreachable");
	}

	var format expr;
	if p.tok == token.COLON {
		p.next();
		format = p.parseExpr();
	}
	
	return &field{name, format};
}


func (p *parser) parseFactor() (x expr) {
	switch p.tok {
	case token.XOR, token.MUL, token.IDENT:
		x = p.parseField();

	case token.STRING:
		x = &literal{p.parseValue()};

	case token.LPAREN:
		p.next();
		x = p.parseExpr();
		p.expect(token.RPAREN);

	case token.LBRACK:
		p.next();
		x = &option{p.parseExpr()};
		p.expect(token.RBRACK);

	case token.LBRACE:
		p.next();
		x = &repetition{p.parseExpr()};
		p.expect(token.RBRACE);

	default:
		p.error_expected(p.pos, "factor");
		p.next();  // make progress
	}

	return x;
}


func (p *parser) parseTerm() expr {
	x := p.parseFactor();

	for	p.tok == token.XOR ||
		p.tok == token.MUL ||
		p.tok == token.IDENT ||
		p.tok == token.STRING ||
		p.tok == token.LPAREN ||
		p.tok == token.LBRACK ||
		p.tok == token.LBRACE
	{
		y := p.parseFactor();
		x = &sequence{x, y};
	}

	return x;
}


func (p *parser) parseExpr() expr {
	x := p.parseTerm();

	for p.tok == token.OR {
		p.next();
		y := p.parseTerm();
		x = &alternative{x, y};
	}

	return x;
}


func (p *parser) parseProduction() (string, expr) {
	name := p.parseName();
	
	var x expr;
	if p.tok == token.ASSIGN {
		p.next();
		if p.tok == token.SEMICOLON {
			x = &empty{};
		} else {
			x = p.parseExpr();
		}
	}

	p.expect(token.SEMICOLON);

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


func typename(value reflect.Value) string {
	name := value.Type().Name();

	if name != "" {
		return name;
	}

	switch value.Kind() {
	case reflect.ArrayKind: name = "array";
	case reflect.BoolKind: name = "bool";
	case reflect.ChanKind: name = "chan";
	case reflect.DotDotDotKind: name = "...";
	case reflect.FloatKind: name = "float";
	case reflect.Float32Kind: name = "float32";
	case reflect.Float64Kind: name = "float64";
	case reflect.FuncKind: name = "func";
	case reflect.IntKind: name = "int";
	case reflect.Int16Kind: name = "int16";
	case reflect.Int32Kind: name = "int32";
	case reflect.Int64Kind: name = "int64";
	case reflect.Int8Kind: name = "int8";
	case reflect.InterfaceKind: name = "interface";
	case reflect.MapKind: name = "map";
	case reflect.PtrKind: name = "pointer";
	case reflect.StringKind: name = "string";
	case reflect.StructKind: name = "struct";
	case reflect.UintKind: name = "uint";
	case reflect.Uint16Kind: name = "uint16";
	case reflect.Uint32Kind: name = "uint32";
	case reflect.Uint64Kind: name = "uint64";
	case reflect.Uint8Kind: name = "uint8";
	case reflect.UintptrKind: name = "uintptr";
	}
	
	return name;
}


var defaultFormat = &literal{io.StringBytes("%v")};

func (f Format) getFormat(value reflect.Value) expr {
	if format, found := f[typename(value)]; found {
		return format;
	}
	// no format found
	return defaultFormat;
}


// Count the number of printf-style '%' formatters in s.
// The result is 0, 1, or 2 (where 2 stands for 2 or more).
//
func percentCount(s []byte) int {
	n := 0;
	for i := 0; n < 2 && i < len(s); i++ {
		// TODO should not count "%%"'s
		if s[i] == '%' {
			n++;
		}
	}
	return n;
}


func printf(w io.Write, format []byte, value reflect.Value) {
	// TODO this seems a bit of a hack
	if percentCount(format) == 1 {
		// exactly one '%' format specifier - try to use it
		fmt.Fprintf(w, string(format), value.Interface());
	} else {
		// 0 or more then 1 '%' format specifier - ignore them
		w.Write(format);
	}
}


// Returns true if a non-empty field value was found.
func (f Format) print(w io.Write, format expr, value reflect.Value, index int) bool {
	switch t := format.(type) {
	case *empty:
		return true;

	case *alternative:
		// print the contents of the first alternative with a non-empty field
		var buf io.ByteBuffer;
		b := f.print(&buf, t.x, value, index);
		if !b {
			b = f.print(&buf, t.y, value, index);
		}
		if b {
			w.Write(buf.Data());
		}
		return index < 0 || b;

	case *sequence:
		b1 := f.print(w, t.x, value, index);
		b2 := f.print(w, t.y, value, index);
		return index < 0 || b1 && b2;

	case *field:
		var x reflect.Value;
		switch t.name {
		case "^":
			if v, is_ptr := value.(reflect.PtrValue); is_ptr {
				if v.Get() == nil {
					return false;
				}
				x = v.Sub();
			} else if v, is_array := value.(reflect.ArrayValue); is_array {
				if index < 0 || v.Len() <= index {
					return false;
				}
				x = v.Elem(index);
			} else if v, is_interface := value.(reflect.InterfaceValue); is_interface {
				if v.Get() == nil {
					return false;
				}
				x = v.Value();
			} else {
				panic("not a ptr, array, or interface");  // TODO fix this
			}
		case "*":
			x = value;
		default:
			if v, is_struct := value.(reflect.StructValue); is_struct {
				x = getField(v, t.name);
			} else {
				panic ("not a struct");  // TODO fix this
			}
		}
		format = t.format;
		if format == nil {
			format = f.getFormat(x);
		}
		b := f.print(w, format, x, index);
		return index < 0 || b;

	case *literal:
		printf(w, t.value, value);
		return true;

	case *option:
		// print the contents of the option if there is a non-empty field
		var buf io.ByteBuffer;
		b := f.print(&buf, t.x, value, -1);
		if b {
			w.Write(buf.Data());
		}
		return index < 0 || b;

	case *repetition:
		// print the contents of the repetition while there is a non-empty field
		b := false;
		for i := 0; ; i++ {
			var buf io.ByteBuffer;
			if f.print(&buf, t.x, value, i) {
				w.Write(buf.Data());
				b = true;
			} else {
				break;
			}
		}
		return index < 0 || b;
		
	case *custom:
		b := t.f(w, value.Interface(), t.name);
		return index < 0 || b;
	}
	
	panic("unreachable");
	return false;
}


func (f Format) Apply(w io.Write, data interface{}) {
	value := reflect.NewValue(data);
	f.print(w, f.getFormat(value), value, -1);
}
