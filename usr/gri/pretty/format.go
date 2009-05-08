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
	"strconv";
	"strings";
)


// TODO should probably do this in a different way
var (
	debug = flag.Bool("d", false, "debug mode");
	trace = flag.Bool("t", false, "trace mode");
)


// ----------------------------------------------------------------------------
// Format representation

type (
	Formatter func(w io.Writer, value interface{}, name string) bool;
	FormatterMap map[string]Formatter;
)


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

	literal struct {
		// TODO should there be other types or should it all be string literals?
		value []byte;
	};

	indentation struct {
		iexpr expr;  // outdent if nil
	};

	field struct {
		fname string;  // including "^", "*"
		tname string;  // "" if no tname specified
	};

	negation struct {
		neg expr;
	};

	option struct {
		opt expr;
	};

	repetition struct {
		rep expr;
		div expr;
	};

	custom struct {
		name string;
		form Formatter
	};
)


func (x *alternative) String() string {
	return fmt.Sprintf("(%v | %v)", x.x, x.y);
}


func (x *sequence) String() string {
	return fmt.Sprintf("%v %v", x.x, x.y);
}


func (x *literal) String() string {
	return strconv.Quote(string(x.value));
}


func (x *indentation) String() string {
	if x.iexpr != nil {
		fmt.Sprintf(">> %s", x.iexpr);
	}
	return "<<";
}


func (x *field) String() string {
	if x.tname == "" {
		return x.fname;
	}
	return x.fname + " : " + x.tname;
}


func (x *negation) String() string {
	return fmt.Sprintf("!%v", x.neg);
}


func (x *option) String() string {
	return fmt.Sprintf("[%v]", x.opt);
}


func (x *repetition) String() string {
	if x.div == nil {
		return fmt.Sprintf("{%v}", x.rep);
	}
	return fmt.Sprintf("{%v / %v}", x.rep, x.div);
}


func (x *custom) String() string {
	return "<" + x.name + ">";
}


/*	A Format is a set of production expressions. A new format is
	created explicitly by calling Parse, or implicitly by one of
	the Xprintf functions.

	Formatting rules are specified in the following syntax:

		Format      = Production { ";" Production } [ ";" ] .
		Production  = Name "=" Expression .
		Name        = identifier { "." identifier } .
		Expression  = [ Term ] { "|" [ Term ] } .
		Term        = Factor { Factor } .
		Factor      = string_literal | Indent | Field | Negation | Group | Option | Repetition .
		Indent      = ">>" Factor | "<<" .
		Field       = ( "^" | "*" | Name ) [ ":" Name ] .
		Negation    = "!" Factor .
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
	- have a format to select type name, field tag, field offset?
	- use field tag as default format for that field
*/

type parser struct {
	// scanning
	scanner scanner.Scanner;
	pos token.Position;  // token position
	tok token.Token;  // one token look-ahead
	lit []byte;  // token literal

	// error handling
	errors io.ByteBuffer;  // errors.Len() > 0 if there were errors
	lastline int;
}


// The parser implements the scanner.ErrorHandler interface.
func (p *parser) Error(pos token.Position, msg string) {
	if pos.Line != p.lastline {
		// only report error if not on the same line as previous error
		// in the hope to reduce number of follow-up errors reported
		fmt.Fprintf(&p.errors, "%d:%d: %s\n", pos.Line, pos.Column, msg);
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


func (p *parser) parseFactor() (x expr)
func (p *parser) parseExpr() expr

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
		x = &indentation{p.parseFactor()};

	case token.SHL:
		p.next();
		x = &indentation{nil};

	case token.NOT:
		p.next();
		x = &negation{p.parseFactor()};

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
		x = p.parseExpr();
		var div expr;
		if p.tok == token.QUO {
			p.next();
			div = p.parseExpr();
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


func (p *parser) parseExpr() expr {
	x := p.parseTerm();

	for p.tok == token.OR {
		p.next();
		y := p.parseTerm();
		x = &alternative{x, y};
	}

	return x;
}


func (p *parser) parseProd() (string, expr) {
	name := p.parseName();
	p.expect(token.ASSIGN);
	x := p.parseExpr();
	return name, x;
}


func (p *parser) parseFormat() Format {
	format := make(Format);

	for p.tok != token.EOF {
		pos := p.pos;
		name, x := p.parseProd();

		// add production to format
		if t, found := format[name]; !found {
			format[name] = x;
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

	return format;
}


type formatError string

func (p formatError) String() string {
	return string(p);
}


func readSource(src interface{}) ([]byte, os.Error) {
	if src == nil {
		return nil, formatError("src is nil");
	}

	switch s := src.(type) {
	case string:
		return io.StringBytes(s), nil;

	case []byte:
		if s == nil {
			return nil, formatError("src is nil");
		}
		return s, nil;

	case *io.ByteBuffer:
		// is io.Read, but src is already available in []byte form
		if s == nil {
			return nil, formatError("src is nil");
		}
		return s.Data(), nil;

	case io.Reader:
		var buf io.ByteBuffer;
		n, err := io.Copy(s, &buf);
		if err != nil {
			return nil, err;
		}
		return buf.Data(), nil
	}

	return nil, formatError("src type not supported");
}


// Parse parses a set of format productions. The format src may be
// a string, a []byte, or implement io.Read. The result is a Format
// if no errors occured; otherwise Parse returns nil.
//
func Parse(src interface{}, fmap FormatterMap) (f Format, err os.Error) {
	s, err := readSource(src);
	if err != nil {
		return nil, err;
	}

	// parse format description
	var p parser;
	p.scanner.Init(s, &p, false);
	p.next();
	f = p.parseFormat();

	// add custom formatters, if any
	for name, form := range fmap {
		if t, found := f[name]; !found {
			f[name] = &custom{name, form};
		} else {
			fmt.Fprintf(&p.errors, "formatter already declared: %s", name);
		}
	}

	if p.errors.Len() > 0 {
		return nil, formatError(string(p.errors.Data()));
	}

	return f, nil;
}


func ParseOrDie(src interface{}, fmap FormatterMap) Format {
	f, err := Parse(src, fmap);
	if err != nil {
		panic(err.String());
	}
	return f;
}


func (f Format) Dump() {
	for name, form := range f {
		fmt.Printf("%s = %v;\n", name, form);
	}
}


// ----------------------------------------------------------------------------
// Formatting

func getField(v reflect.StructValue, fieldname string) reflect.Value {
	t := v.Type().(reflect.StructType);
	for i := 0; i < t.Len(); i++ {
		name, typ, tag, offset := t.Field(i);
		if name == fieldname {
			return v.Field(i);
		} else if name == "" {
			// anonymous field - check type name
			// TODO this is only going down one level - fix
			if strings.HasSuffix(typ.Name(), "." + fieldname) {
				return v.Field(i);
			}
		}
	}
	panicln(fmt.Sprintf("no field %s int %s", fieldname, t.Name()));
	return nil;
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
	case reflect.DotDotDotKind: name = "ellipsis";
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
	reflect.ArrayKind: &field{"*", ""},
	reflect.DotDotDotKind: &field{"*", ""},
	reflect.InterfaceKind: &field{"*", ""},
	reflect.MapKind: &field{"*", ""},
	reflect.PtrKind: &field{"*", ""},
	reflect.StringKind: &literal{io.StringBytes("%s")},
}

var catchAll = &literal{io.StringBytes("%v")};

func (f Format) getFormat(name string, value reflect.Value) expr {
	/*
	if name == "nil" {
		fmt.Printf("value = %T %v, kind = %d\n", value, value, value.Kind());
		panic();
	}
	*/

	if fexpr, found := f[name]; found {
		return fexpr;
	}

	if *debug {
		fmt.Printf("no production for type: %s\n", name);
	}

	// no fexpr found - return kind-specific default value, if any
	if fexpr, found := defaults[value.Kind()]; found {
		return fexpr;
	}

	if *debug {
		fmt.Printf("no default for type: %s\n", name);
	}

	return catchAll;
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


func rawPrintf(w io.Writer, format []byte, value reflect.Value) {
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


// TODO this should become a Go built-in
func push(dst []int, x int) []int {
	n := len(dst);
	if n > cap(dst) {
		panic("dst too small");
	}
	dst = dst[0 : n+1];
	dst[n] = x;
	return dst;
}


func append(dst, src []byte) []byte {
	n, m := len(dst), len(src);
	if n+m > cap(dst) {
		panic("dst too small");
	}
	dst = dst[0 : n+m];
	for i := 0; i < m; i++ {
		dst[n+i] = src[i];
	}
	return dst;
}


type state struct {
	f Format;

	// indentation
	indent_text []byte;
	indent_widths []int;
}


func (ps *state) init(f Format) {
	ps.f = f;
	ps.indent_text = make([]byte, 0, 1000);  // TODO don't use fixed cap
	ps.indent_widths = make([]int, 0, 100);  // TODO don't use fixed cap
}


func (ps *state) indent(text []byte) {
	ps.indent_widths = push(ps.indent_widths, len(ps.indent_text));
	ps.indent_text = append(ps.indent_text, text);
}


func (ps *state) outdent() {
	i := len(ps.indent_widths);
	if i > 0 {
		ps.indent_text = ps.indent_text[0 : ps.indent_widths[i-1]];
		ps.indent_widths = ps.indent_widths[0 : i-1];
	}
}


func (ps *state) printIndented(w io.Writer, s []byte) {
	// replace each '\n' with the indent + '\n'
	i0 := 0;
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			w.Write(s[i0 : i+1]);
			w.Write(ps.indent_text);
			i0 = i+1;
		}
	}
	w.Write(s[i0 : len(s)]);
}


func (ps *state) printf(w io.Writer, format []byte, value reflect.Value) {
	if len(ps.indent_widths) == 0 {
		// no indentation
		rawPrintf(w, format, value);
	} else {
		// print into temporary buffer
		var buf io.ByteBuffer;
		rawPrintf(&buf, format, value);
		ps.printIndented(w, buf.Data());
	}
}


func (ps *state) print(w io.Writer, fexpr expr, value reflect.Value, index, level int) bool

// Returns true if a non-empty field value was found.
func (ps *state) print0(w io.Writer, fexpr expr, value reflect.Value, index, level int) bool {
	if fexpr == nil {
		return true;
	}

	switch t := fexpr.(type) {
	case *alternative:
		// - print the contents of the first alternative with a non-empty field
		// - result is true if there is at least one non-empty field
		var buf io.ByteBuffer;
		if ps.print(&buf, t.x, value, 0, level) {
			w.Write(buf.Data());
			return true;
		} else {
			var buf io.ByteBuffer;
			if ps.print(&buf, t.y, value, 0, level) {
				w.Write(buf.Data());
				return true;
			}
		}
		return false;

	case *sequence:
		// - print the contents of the sequence
		// - result is true if there is no empty field
		// TODO do we need to buffer here? why not?
		b1 := ps.print(w, t.x, value, index, level);
		b2 := ps.print(w, t.y, value, index, level);
		return b1 && b2;

	case *literal:
		// - print the literal
		// - result is always true (literal is never empty)
		ps.printf(w, t.value, value);
		return true;

	case *indentation:
		if t.iexpr != nil {
			// indent
			var buf io.ByteBuffer;
			ps.print(&buf, t.iexpr, value, index, level);
			ps.indent(buf.Data());

		} else {
			// outdent
			ps.outdent();
		}
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
				panic("reflection support for maps incomplete");

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
				panic(fmt.Sprintf("error: * does not apply to `%s`\n", value.Type().Name()));
			}

		default:
			// field
			if s, is_struct := value.(reflect.StructValue); is_struct {
				value = getField(s, t.fname);
			} else {
				// TODO fix this
				panic(fmt.Sprintf("error: %s has no field `%s`\n", value.Type().Name(), t.fname));
			}
		}

		// determine format
		tname := t.tname;
		if tname == "" {
			tname = typename(value)
		}
		fexpr = ps.f.getFormat(tname, value);

		return ps.print(w, fexpr, value, index, level);

	case *negation:
		// TODO is this operation useful at all?
		// print the contents of the option if is contains an empty field
		var buf io.ByteBuffer;
		if !ps.print(&buf, t.neg, value, 0, level) {
			w.Write(buf.Data());
		}
		return true;

	case *option:
		// print the contents of the option if it contains a non-empty field
		var buf io.ByteBuffer;
		if ps.print(&buf, t.opt, value, 0, level) {
			w.Write(buf.Data());
		}
		return true;

	case *repetition:
		// print the contents of the repetition while there is a non-empty field
		var buf io.ByteBuffer;
		for i := 0; ps.print(&buf, t.rep, value, i, level); i++ {
			if i > 0 {
				ps.print(w, t.div, value, i, level);
			}
			w.Write(buf.Data());
			buf.Reset();
		}
		return true;

	case *custom:
		var buf io.ByteBuffer;
		if t.form(&buf, value.Interface(), t.name) {
			ps.printIndented(w, buf.Data());
			return true;
		}
		return false;
	}

	panic("unreachable");
	return false;
}


func printTrace(indent int, format string, a ...) {
	const dots =
		". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
		". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ";
	const n = len(dots);
	i := 2*indent;
	for ; i > n; i -= n {
		fmt.Print(dots);
	}
	fmt.Print(dots[0 : i]);
	fmt.Printf(format, a);
}


func (ps *state) print(w io.Writer, fexpr expr, value reflect.Value, index, level int) bool {
	if *trace {
		printTrace(level, "%v, %d {\n", fexpr, /*value.Interface(), */index);
	}

	result := ps.print0(w, fexpr, value, index, level+1);

	if *trace {
		printTrace(level, "} %v\n", result);
	}
	return result;
}


// TODO proper error reporting

// Fprint formats each argument according to the format f
// and writes to w.
//
func (f Format) Fprint(w io.Writer, args ...) {
	value := reflect.NewValue(args).(reflect.StructValue);
	for i := 0; i < value.Len(); i++ {
		fld := value.Field(i);
		var ps state;
		ps.init(f);
		ps.print(w, f.getFormat(typename(fld), fld), fld, 0, 0);
	}
}


// Print formats each argument according to the format f
// and writes to standard output.
//
func (f Format) Print(args ...) {
	f.Fprint(os.Stdout, args);
}


// Sprint formats each argument according to the format f
// and returns the resulting string.
//
func (f Format) Sprint(args ...) string {
	var buf io.ByteBuffer;
	f.Fprint(&buf, args);
	return string(buf.Data());
}
