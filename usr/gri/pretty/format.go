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

	When printing a value, its type name is used to look up the production
	to be printed. Literal values are printed as is, field references are
	resolved and the respective field values are printed instead (using their
	type-specific productions), and alternative, grouped, optional, and
	repetitive sub-expressions are printed depending on whether they contain
	"empty" fields or not. A field is empty if its value is nil.
*/
package format

import (
	"flag";
	"fmt";
	"go/scanner";
	"go/token";
	"io";
	"os";
	"reflect";
	"runtime";
	"strconv";
	"strings";
)


// ----------------------------------------------------------------------------
// Format representation

type (
	Formatter func(w io.Writer, env, value interface{}, name string) bool;
	FormatterMap map[string]Formatter;
)


// A production expression is built from the following nodes.
//
type (
	expr interface {};

	alternative struct {
		x, y expr;  // x | y
	};

	sequence struct {
		x, y expr;  // x y
	};

	literal struct {
		value []byte;
	};

	field struct {
		fname string;  // including "^", "*"
		tname string;  // "" if no tname specified
	};

	indentation struct {
		indent, body expr;  // >> indent body <<
	};

	option struct {
		body expr;  // [body]
	};

	repetition struct {
		body, div expr;  // {body / div}
	};

	custom struct {
		name string;
		form Formatter
	};
)


/*	A Format is a set of production expressions. A new format is
	created explicitly by calling Parse, or implicitly by one of
	the Xprintf functions.

	Formatting rules are specified in the following syntax:

		Format      = Production { ";" Production } [ ";" ] .
		Production  = ( Name | "default" | "/" ) "=" Expression .
		Name        = identifier { "." identifier } .
		Expression  = [ Term ] { "|" [ Term ] } .
		Term        = Factor { Factor } .
		Factor      = string_literal | Indent | Field | Group | Option | Repetition .
		Field       = ( "^" | "*" | Name ) [ ":" Name ] .
		Indent      = ">>" Factor Expression "<<" .
		Group       = "(" Expression ")" .
		Option      = "[" Expression "]" .
		Repetition  = "{" Expression [ "/" Expression ] "}" .

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

		: Name

	which specifies the field format rule irrespective of the field type.

	TODO complete this description
*/
type Format struct {
	// TODO(gri) Eventually have import path info here
	//           once reflect provides import paths.
	rules map [string] expr;
}



// ----------------------------------------------------------------------------
// Error handling

// Error implements an os.Error that may be returned as a
// result of calling Parse or any of the print functions.
//
type Error struct {
	Pos token.Position;  // source position, if any (otherwise Pos.Line == 0)
	Msg string;  // error message
	Next *Error;  // next error, if any (or nil)
}


// String converts a list of Error messages into a string,
// with one error per line.
//
func (e *Error) String() string {
	var buf io.ByteBuffer;
	for ; e != nil; e = e.Next {
		if e.Pos.Line > 0 {
			fmt.Fprintf(&buf, "%d:%d: ", e.Pos.Line, e.Pos.Column);
		}
		fmt.Fprintf(&buf, "%s\n", e.Msg);
	}
	return string(buf.Data());
}


// ----------------------------------------------------------------------------
// Parsing

/*	TODO
	- have a format to select type name, field tag, field offset?
	- use field tag as default format for that field
*/

type parser struct {
	// scanning
	scanner scanner.Scanner;
	pos token.Position;  // token position
	tok token.Token;  // one token look-ahead
	lit []byte;  // token literal

	// errors
	first, last *Error;
}


// The parser implements the scanner.ErrorHandler interface.
func (p *parser) Error(pos token.Position, msg string) {
	if p.last == nil || p.last.Pos.Line != pos.Line {
		// only report error if not on the same line as previous error
		// in the hope to reduce number of follow-up errors reported
		err := &Error{pos, msg, nil};
		if p.last == nil {
			p.first = err;
		} else {
			p.last.Next = err;
		}
		p.last = err;
	}
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


func (p *parser) parseValue() []byte {
	if p.tok != token.STRING {
		p.expect(token.STRING);
		return nil;  // TODO should return something else?
	}

	// TODO get rid of back-and-forth conversions
	//      (change value to string?)
	s, err := strconv.Unquote(string(p.lit));
	if err != nil {
		panic("scanner error");
	}

	p.next();
	return io.StringBytes(s);
}


func (p *parser) parseFactor() (x expr)
func (p *parser) parseExpression() expr

func (p *parser) parseField() expr {
	var fname string;
	switch p.tok {
	case token.XOR:
		fname = "^";
		p.next();
	case token.MUL:
		fname = "*";
		p.next();
	case token.IDENT:
		// TODO use reflect.ExpandType() to lookup a field
		// during parse-time if posssible
		fname = p.parseName();
	default:
		return nil;
	}

	var tname string;
	if p.tok == token.COLON {
		p.next();
		tname = p.parseName();
	}

	return &field{fname, tname};
}


func (p *parser) parseFactor() (x expr) {
	switch p.tok {
	case token.STRING:
		x = &literal{p.parseValue()};

	case token.SHR:
		p.next();
		iexpr := p.parseFactor();
		body := p.parseExpression();
		p.expect(token.SHL);
		return &indentation{iexpr, body};

	case token.LPAREN:
		p.next();
		x = p.parseExpression();
		p.expect(token.RPAREN);

	case token.LBRACK:
		p.next();
		x = &option{p.parseExpression()};
		p.expect(token.RBRACK);

	case token.LBRACE:
		p.next();
		x = p.parseExpression();
		var div expr;
		if p.tok == token.QUO {
			p.next();
			div = p.parseExpression();
		}
		x = &repetition{x, div};
		p.expect(token.RBRACE);

	default:
		x = p.parseField();
	}

	return x;
}


func (p *parser) parseTerm() expr {
	x := p.parseFactor();

	if x != nil {
		for {
			y := p.parseFactor();
			if y == nil {
				break;
			}
			x = &sequence{x, y};
		}
	}

	return x;
}


func (p *parser) parseExpression() expr {
	x := p.parseTerm();

	for p.tok == token.OR {
		p.next();
		y := p.parseTerm();
		x = &alternative{x, y};
	}

	return x;
}


func (p *parser) parseProduction() (string, expr) {
	var name string;
	switch p.tok {
	case token.DEFAULT:
		p.next();
		name = "default";
	case token.QUO:
		p.next();
		name = "/";
	default:
		name = p.parseName();
	}
	p.expect(token.ASSIGN);
	x := p.parseExpression();
	return name, x;
}


func (p *parser) parseFormat() *Format {
	rules := make(map [string] expr);

	for p.tok != token.EOF {
		pos := p.pos;
		name, x := p.parseProduction();

		// add production to rules
		if t, found := rules[name]; !found {
			rules[name] = x;
		} else {
			p.Error(pos, "production already declared: " + name);
		}

		if p.tok == token.SEMICOLON {
			p.next();
		} else {
			break;
		}
	}
	p.expect(token.EOF);

	return &Format{rules};
}


// Parse parses a set of format productions from source src. If there are no
// errors, the result is a Format and the error is nil. Otherwise the format
// is nil and the os.Error string contains a line for each error encountered.
//
func Parse(src []byte, fmap FormatterMap) (*Format, os.Error) {
	// parse source
	var p parser;
	p.scanner.Init(src, &p, false);
	p.next();
	f := p.parseFormat();

	// add custom formatters, if any
	// TODO should we test that name is a legal name?
	for name, form := range fmap {
		if t, found := f.rules[name]; !found {
			f.rules[name] = &custom{name, form};
		} else {
			p.Error(token.Position{0, 0, 0}, "formatter already declared: " + name);
		}
	}

	if p.first != nil {
		return nil, p.first;
	}

	return f, nil;
}


// ----------------------------------------------------------------------------
// Formatting

type state struct {
	f *Format;
	env interface{};
	sep expr;
	errors chan os.Error;  // not chan *Error: errors <- nil would be wrong!
	indent io.ByteBuffer;
}


func (ps *state) init(f *Format, env interface{}) {
	ps.f = f;
	ps.env = env;
	// if we have a separator ("/") production, cache it for easy access
	if sep, has_sep := f.rules["/"]; has_sep {
		ps.sep = sep;
	}
	ps.errors = make(chan os.Error);
}


func (ps *state) error(msg string) {
	ps.errors <- &Error{token.Position{0, 0, 0}, msg, nil};
	runtime.Goexit();
}


func getField(val reflect.Value, fieldname string) (reflect.Value, int) {
	// do we have a struct in the first place?
	if val.Kind() != reflect.StructKind {
		return nil, 0;
	}
	
	sval, styp := val.(reflect.StructValue), val.Type().(reflect.StructType);

	// look for field at the top level
	for i := 0; i < styp.Len(); i++ {
		name, typ, tag, offset := styp.Field(i);
		if name == fieldname || name == "" && strings.HasSuffix(typ.Name(), "." + fieldname) /* anonymous field */ {
			return sval.Field(i), 0;
		}
	}

	// look for field in anonymous fields
	var field reflect.Value;
	level := 1000;  // infinity
	for i := 0; i < styp.Len(); i++ {
		name, typ, tag, offset := styp.Field(i);
		if name == "" {
			f, l := getField(sval.Field(i), fieldname);
			// keep the most shallow field
			if f != nil && l < level {
				field, level = f, l;
			}
		}
	}
	
	return field, level + 1;
}


var default_names = map[int]string {
	reflect.ArrayKind: "array",
	reflect.BoolKind: "bool",
	reflect.ChanKind: "chan",
	reflect.DotDotDotKind: "ellipsis",
	reflect.FloatKind: "float",
	reflect.Float32Kind: "float32",
	reflect.Float64Kind: "float64",
	reflect.FuncKind: "func",
	reflect.IntKind: "int",
	reflect.Int16Kind: "int16",
	reflect.Int32Kind: "int32",
	reflect.Int64Kind: "int64",
	reflect.Int8Kind: "int8",
	reflect.InterfaceKind: "interface",
	reflect.MapKind: "map",
	reflect.PtrKind: "pointer",
	reflect.StringKind: "string",
	reflect.StructKind: "struct",
	reflect.UintKind: "uint",
	reflect.Uint16Kind: "uint16",
	reflect.Uint32Kind: "uint32",
	reflect.Uint64Kind: "uint64",
	reflect.Uint8Kind: "uint8",
	reflect.UintptrKind: "uintptr",
}


func typename(value reflect.Value) string {
	name := value.Type().Name();
	if name == "" {
		if default_name, found := default_names[value.Kind()]; found {
			name = default_name;
		}
	}
	return name;
}


func (ps *state) getFormat(name string) expr {
	if fexpr, found := ps.f.rules[name]; found {
		return fexpr;
	}

	if fexpr, found := ps.f.rules["default"]; found {
		return fexpr;
	}

	ps.error(fmt.Sprintf("no production for type: '%s'\n", name));
	panic("unreachable");
	return nil;
}


// Count the number of printf-style '%' formatters in s.
//
func percentCount(s []byte) int {
	n := 0;
	for i := 0; i < len(s); i++ {
		if s[i] == '%' {
			i++;
			if i >= len(s) || s[i] != '%' {  // don't count "%%"
				n++;
			}
		}
	}
	return n;
}


func (ps *state) rawPrintf(w io.Writer, format []byte, value reflect.Value) {
	// TODO find a better way to do this
	x := value.Interface();
	switch percentCount(format) {
	case  0: w.Write(format);
	case  1: fmt.Fprintf(w, string(format), x);
	case  2: fmt.Fprintf(w, string(format), x, x);
	case  3: fmt.Fprintf(w, string(format), x, x, x);
	case  4: fmt.Fprintf(w, string(format), x, x, x, x);
	default: panic("no support for more than 4 '%'-format chars yet");
	}
}


func (ps *state) printIndented(w io.Writer, s []byte) {
	// replace each '\n' with the indent + '\n'
	i0 := 0;
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			w.Write(s[i0 : i+1]);
			w.Write(ps.indent.Data());
			i0 = i+1;
		}
	}
	w.Write(s[i0 : len(s)]);
}


func (ps *state) printf(w io.Writer, format []byte, value reflect.Value) {
	if ps.indent.Len()== 0 {
		// no indentation
		ps.rawPrintf(w, format, value);
	} else {
		// print into temporary buffer
		var buf io.ByteBuffer;
		ps.rawPrintf(&buf, format, value);
		ps.printIndented(w, buf.Data());
	}
}


// Returns true if a non-empty field value was found.
func (ps *state) print(w io.Writer, fexpr expr, value reflect.Value, index int) bool {
	if fexpr == nil {
		return true;
	}

	switch t := fexpr.(type) {
	case *alternative:
		// - print the contents of the first alternative with a non-empty field
		// - result is true if there is at least one non-empty field
		var buf io.ByteBuffer;
		if ps.print(&buf, t.x, value, 0) {
			w.Write(buf.Data());
			return true;
		} else {
			var buf io.ByteBuffer;
			if ps.print(&buf, t.y, value, 0) {
				w.Write(buf.Data());
				return true;
			}
		}
		return false;

	case *sequence:
		// - print the contents of the sequence
		// - result is true if there is no empty field
		// TODO do we need to buffer here? why not?
		b := ps.print(w, t.x, value, index);
		// TODO should invoke separator only inbetween terminal symbols?
		if ps.sep != nil {
			b = ps.print(w, ps.sep, value, index) && b;
		}
		return ps.print(w, t.y, value, index) && b;

	case *literal:
		// - print the literal
		// - result is always true (literal is never empty)
		ps.printf(w, t.value, value);
		return true;

	case *field:
		// - print the contents of the field
		// - format is either the field format or the type-specific format
		// - TODO look at field tag for default format
		// - result is true if the field is not empty
		switch t.fname {
		case "^":
			// identity - value doesn't change

		case "*":
			// indirect
			switch v := value.(type) {
			case reflect.ArrayValue:
				if v.Len() <= index {
					return false;
				}
				value = v.Elem(index);

			case reflect.MapValue:
				ps.error("reflection support for maps incomplete\n");

			case reflect.PtrValue:
				if v.Get() == nil {
					return false;
				}
				value = v.Sub();

			case reflect.InterfaceValue:
				if v.Get() == nil {
					return false;
				}
				value = v.Value();

			default:
				// TODO fix this
				ps.error(fmt.Sprintf("error: * does not apply to `%s`\n", value.Type().Name()));
			}

		default:
			// field
			field, _ := getField(value, t.fname);
			if field == nil {
				ps.error(fmt.Sprintf("error: no field `%s` in `%s`\n", t.fname, value.Type().Name()));
			}
			value = field;
		}

		// determine format
		tname := t.tname;
		if tname == "" {
			tname = typename(value)
		}
		fexpr = ps.getFormat(tname);

		return ps.print(w, fexpr, value, index);

	case *indentation:
		saved_len := ps.indent.Len();
		ps.print(&ps.indent, t.indent, value, index);  // add additional indentation
		b := ps.print(w, t.body, value, index);
		ps.indent.Truncate(saved_len);  // reset indentation
		return b;

	case *option:
		// print the contents of the option if it contains a non-empty field
		var buf io.ByteBuffer;
		if ps.print(&buf, t.body, value, 0) {
			w.Write(buf.Data());
		}
		return true;

	case *repetition:
		// print the contents of the repetition while there is a non-empty field
		var buf io.ByteBuffer;
		for i := 0; ps.print(&buf, t.body, value, i); i++ {
			if i > 0 {
				ps.print(w, t.div, value, i);
			}
			w.Write(buf.Data());
			buf.Reset();
		}
		return true;

	case *custom:
		var buf io.ByteBuffer;
		if t.form(&buf, ps.env, value.Interface(), t.name) {
			ps.printIndented(w, buf.Data());
			return true;
		}
		return false;
	}

	panic("unreachable");
	return false;
}


// Fprint formats each argument according to the format f
// and writes to w.
//
func (f *Format) Fprint(w io.Writer, env interface{}, args ...) (int, os.Error) {
	var ps state;
	ps.init(f, env);

	go func() {
		value := reflect.NewValue(args).(reflect.StructValue);
		for i := 0; i < value.Len(); i++ {
			fld := value.Field(i);
			ps.print(w, ps.getFormat(typename(fld)), fld, 0);
		}
		ps.errors <- nil;  // no errors
	}();

	// TODO return correct value for count instead of 0
	return 0, <-ps.errors;
}


// Print formats each argument according to the format f
// and writes to standard output.
//
func (f *Format) Print(args ...) (int, os.Error) {
	return f.Fprint(os.Stdout, nil, args);
}


// Sprint formats each argument according to the format f
// and returns the resulting string.
//
func (f *Format) Sprint(args ...) string {
	var buf io.ByteBuffer;
	// TODO what to do in case of errors?
	f.Fprint(&buf, nil, args);
	return string(buf.Data());
}
