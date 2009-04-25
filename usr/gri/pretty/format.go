// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*	The format package implements syntax-directed formatting of arbitrary
	data structures.

	A format specification consists of a set of named productions in EBNF.
	The production names correspond to the type names of the data structure
	to be printed. The production expressions consist of literal values
	(strings), references to fields, and alternative, grouped, optional,
	and repetitive sub-expressions.

	When printing a value, its type name is used to lookup the production
	to be printed. Literal values are printed as is, field references are
	resolved and the respective field value is printed instead (using its
	type-specific production), and alternative, grouped, optional, and
	repetitive sub-expressions are printed depending on whether they contain
	"empty" fields or not. A field is empty if its value is nil.
*/
package format

import (
	"fmt";
	"go/scanner";
	"go/token";
	"io";
	"os";
	"reflect";
	"strconv";
)


// ----------------------------------------------------------------------------
// Format representation

// A production expression is built from the following nodes.
//
type (
	expr interface {
		String() string;
	};

	alternative struct {
		x, y expr;
	};

	sequence struct {
		x, y expr;
	};

	field struct {
		name string;  // including "^", "*"
		fexpr expr;  // nil if no fexpr specified
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


// TODO If we had a basic accessor mechanism in the language (a field
// "f T" automatically implements a corresponding accessor "f() T", this
// could be expressed more easily by simply providing the field.
//

func (x *alternative) String() string {
	return fmt.Sprintf("(%v | %v)", x.x, x.y);
}


func (x *sequence) String() string {
	return fmt.Sprintf("%v %v", x.x, x.y);
}


func (x *field) String() string {
	if x.fexpr == nil {
		return x.name;
	}
	return fmt.Sprintf("%s: (%v)", x.name, x.fexpr);
}


func (x *literal) String() string {
	return strconv.Quote(string(x.value));
}


func (x *option) String() string {
	return fmt.Sprintf("[%v]", x.x);
}


func (x *repetition) String() string {
	return fmt.Sprintf("{%v}", x.x);
}


func (x *custom) String() string {
	return fmt.Sprintf("<custom %s>", x.name);
}


/*	A Format is a set of production expressions. A new format is
	created explicitly by calling Parse, or implicitly by one of
	the Xprintf functions.

	Formatting rules are specified in the following syntax:

		Format      = { Production } .
		Production  = Name [ "=" [ Expression ] ] ";" .
		Name        = identifier { "." identifier } .
		Expression  = Term { "|" Term } .
		Term        = Factor { Factor } .
		Factor      = string_literal | Field | Group | Option | Repetition .
		Field		= ( "^" | "*" | Name ) [ ":" Expression ] .
		Group       = "(" Expression ")" .
		Option      = "[" Expression "]" .
		Repetition  = "{" Expression "}" .

	The syntax of white space, comments, identifiers, and string literals is
	the same as in Go.
	
	A production name corresponds to a Go type name of the form

		PackageName.TypeName

	(for instance format.Format). A production of the form
	
		Name;

	specifies a package name which is prepended to all subsequent production
	names:

		format;
		Format = ...	// this production matches the type format.Format

	The basic operands of productions are string literals, field names, and
	designators. String literals are printed as is, unless they contain a
	single %-style format specifier (such as "%d"). In that case, they are
	used as the format for fmt.Printf, with the current value as argument.

	The designator "^" stands for the current value; a "*" denotes indirection
	(pointers, arrays, maps, and interfaces).

	A field may contain a format specifier of the form

		: Expression

	which specifies the field format irrespective of the field type.

	Default formats are used for types without specific formating rules:
	The "%v" format is used for values of all types expect pointer, array,
	map, and interface types. They are using the "^" designator.

	TODO complete this description
*/
type Format map [string] expr;


// ----------------------------------------------------------------------------
// Parsing

/*	TODO
	- installable custom formatters (like for template.go)
	- have a format to select type name, field tag, field offset?
	- use field tag as default format for that field
	- field format override (":") is not working as it should
	  (cannot refer to another production - syntactially not possible
	  at the moment)
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


func (p *parser) parseValue() []byte {
	if p.tok != token.STRING {
		p.expect(token.STRING);
		return nil;  // TODO should return something else?
	}

	// TODO get rid of back-and-forth conversions
	//      (change value to string?)
	s, err := strconv.Unquote(string(p.lit));
	if err != nil {
		panic("scanner error?");
	}
	
	p.next();
	return io.StringBytes(s);
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
		return nil;
	}

	var fexpr expr;
	if p.tok == token.COLON {
		p.next();
		fexpr = p.parseExpr();
	}
	
	return &field{name, fexpr};
}


func (p *parser) parseFactor() (x expr) {
	switch p.tok {
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
		x = p.parseField();
	}

	return x;
}


func (p *parser) parseTerm() expr {
	x := p.parseFactor();
	if x == nil {
		p.error_expected(p.pos, "factor");
		p.next();  // make progress
		return nil;
	}

	for {
		y := p.parseFactor();
		if y == nil {
			break;
		}
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


func (p *parser) parseFormat() Format {
	format := make(Format);
	
	prefix := "";
	for p.tok != token.EOF {
		pos := p.pos;
		name := p.parseName();
		
		if p.tok == token.ASSIGN {
			// production
			p.next();
			var x expr;
			if p.tok != token.SEMICOLON {
				x = p.parseExpr();
			}
			// add production to format
			name = prefix + name;
			if t, found := format[name]; !found {
				format[name] = x;
			} else {
				p.Error(pos, "production already declared: " + name);
			}
			
		} else {
			// prefix only
			prefix = name + ".";
		}
		
		p.expect(token.SEMICOLON);
	}
	
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


// TODO do better error handling

// Parse parses a set of format productions. The format src may be
// a string, a []byte, or implement io.Read. The result is a Format
// if no errors occured; otherwise Parse returns nil.
//
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


// ----------------------------------------------------------------------------
// Formatting

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


var defaults = map [int] expr {
	reflect.ArrayKind: &field{"*", nil},
	reflect.MapKind: &field{"*", nil},
	reflect.PtrKind: &field{"*", nil},
}

var catchAll = &literal{io.StringBytes("%v")};

func (f Format) getFormat(value reflect.Value) expr {
	if fexpr, found := f[typename(value)]; found {
		return fexpr;
	}

	// no fexpr found - return kind-specific default value, if any
	if fexpr, found := defaults[value.Kind()]; found {
		return fexpr;
	}

	return catchAll;
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
func (f Format) print(w io.Write, fexpr expr, value reflect.Value, index int) bool {
	debug := false;  // enable for debugging
	if debug {
		fmt.Printf("print(%v, = %v, %v, %d)\n", w, fexpr, value.Interface(), index);
	}

	if fexpr == nil {
		return true;
	}

	switch t := fexpr.(type) {
	case *alternative:
		// - print the contents of the first alternative with a non-empty field
		// - result is true if there is at least one non-empty field
		b := false;
		var buf io.ByteBuffer;
		if f.print(&buf, t.x, value, index) {
			w.Write(buf.Data());
			b = true;
		} else {
			buf.Reset();
			if f.print(&buf, t.y, value, 0) {
				w.Write(buf.Data());
				b = true;
			}
		}
		return b;

	case *sequence:
		// - print the contents of the sequence
		// - result is true if there is no empty field
		// TODO do we need to buffer here? why not?
		b1 := f.print(w, t.x, value, index);
		b2 := f.print(w, t.y, value, index);
		return b1 && b2;

	case *field:
		// - print the contents of the field
		// - format is either the field format or the type-specific format
		// - TODO look at field tag for default format
		// - result is true if the field is not empty
		switch t.name {
		case "^":
			// identity - value doesn't change

		case "*":
			// indirect
			switch v := value.(type) {
			case reflect.PtrValue:
				if v.Get() == nil {
					return false;
				}
				value = v.Sub();

			case reflect.ArrayValue:
				if index < 0 || v.Len() <= index {
					return false;
				}
				value = v.Elem(index);

			case reflect.MapValue:
				panic("reflection support for maps incomplete");

			case reflect.InterfaceValue:
				if v.Get() == nil {
					return false;
				}
				value = v.Value();

			default:
				panic("not a ptr, array, map, or interface");  // TODO fix this
			}

		default:
			// field
			if s, is_struct := value.(reflect.StructValue); is_struct {
				value = getField(s, t.name);
			} else {
				panic ("not a struct");  // TODO fix this
			}
		}

		// determine format
		fexpr = t.fexpr;
		if fexpr == nil {
			// no field format - use type-specific format
			fexpr = f.getFormat(value);
		}

		return f.print(w, fexpr, value, index);
		// BUG (6g?) crash with code below
		/*
		var buf io.ByteBuffer;
		if f.print(&buf, fexpr, value, index) {
			w.Write(buf.Data());
			return true;
		}
		return false;
		*/

	case *literal:
		// - print the literal
		// - result is always true (literal is never empty)
		printf(w, t.value, value);
		return true;

	case *option:
		// print the contents of the option if it contains a non-empty field
		//var foobar bool;  // BUG w/o this declaration the code works!!!
		var buf io.ByteBuffer;
		if f.print(&buf, t.x, value, 0) {
			w.Write(buf.Data());
			return true;
		}
		return false;

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
		return b;
		
	case *custom:
		return t.f(w, value.Interface(), t.name);
	}

	panic("unreachable");
	return false;
}


// TODO proper error reporting

// Fprint formats each argument according to the format f
// and writes to w.
//
func (f Format) Fprint(w io.Write, args ...) {
	value := reflect.NewValue(args).(reflect.StructValue);
	for i := 0; i < value.Len(); i++ {
		fld := value.Field(i);
		f.print(w, f.getFormat(fld), fld, -1);
	}
}


// Fprint formats each argument according to the format f
// and writes to standard output.
//
func (f Format) Print(args ...) {
	f.Print(os.Stdout, args);
}


// Fprint formats each argument according to the format f
// and returns the resulting string.
//
func (f Format) Sprint(args ...) string {
	var buf io.ByteBuffer;
	f.Fprint(&buf, args);
	return string(buf.Data());
}
