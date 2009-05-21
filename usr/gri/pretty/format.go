// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*	The format package implements syntax-directed, type-driven formatting
	of arbitrary data structures. Formatting a data structure consists of
	two phases: first, a format specification is parsed (once per format)
	which results in a "compiled" format. The format can then be used
	repeatedly to print arbitrary values to a io.Writer.

	A format specification consists of a set of named format rules in EBNF.
	The rule names correspond to the type names of the data structure to be
	formatted. Each format rule consists of literal values and struct field
	names which are combined into sequences, alternatives, grouped, optional,
	repeated, or indented sub-expressions. Additionally, format rules may be
	specified via Go formatter functions.

	When formatting a value, its type name determines the format rule. The
	syntax of the rule or the corresponding formatter function determines
	if and how the value is formatted. A format rule may refer to a struct
	field of the current value. In this case the same mechanism is applied
	recursively to that field.
*/
package format

import (
	"container/vector";
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

// Custom formatters implement the Formatter function type.
// A formatter is invoked with a writer w, an environment env
// (provided to format.Fprint and simply passed through), the
// value to format, and the rule name under which the formatter
// was installed (the same formatter function may be installed
// under different names).
//
type Formatter func(w io.Writer, env, value interface{}, rule_name string) bool


// A FormatterMap is a set of custom formatters.
// It maps a rule name to a formatter.
//
type FormatterMap map [string] Formatter;


// A production expression is built from the following nodes.
//
type (
	expr interface {};

	alternatives []expr;  // x | y | z

	sequence []expr;  // x y z

	// a literal is represented as string or []byte

	field struct {
		field_name string;  // including "^", "*"
		rule_name string;  // "" if no rule name specified
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
		rule_name string;
		form Formatter
	};
)


/*	The syntax of a format specification is presented in the same EBNF
	notation as used in the Go language spec. The syntax of white space,
	comments, identifiers, and string literals is the same as in Go.

	A format specification consists of a possibly empty set of package
	declarations and format rules:

		Format      = [ Entry { ";" Entry } ] [ ";" ] .
		Entry       = PackageDecl | FormatRule . 

	A package declaration binds a package name (such as 'ast') to a
	package import path (such as '"go/ast"'). A package name must be
	declared at most once.

		PackageDecl = PackageName ImportPath .
		PackageName = identifier .
		ImportPath  = string .

	A format rule binds a rule name to a format expression. A rule name
	may be a type name or one of the special names 'default' (denoting
	the default rule) or '/' (denoting the global "divider" rule - see
	below). A type name may be the name of a predeclared type ('int',
	'float32', etc.), the name of an anonymous composite type ('array',
	'pointer', etc.), or the name of a user-defined type qualified by
	the corresponding package name (for instance 'ast.MapType'). The
	package name must have been declared already. A rule name must be
	declared at most once.

		FormatRule  = RuleName "=" Expression .
		RuleName    = TypeName | "default" | "/" .
		TypeName    = [ PackageName "." ] identifier .

	A format expression specifies how a value is to be formatted. In its
	most general form, a format expression is a set of alternatives separated
	by "|". Each alternative and the entire expression may be empty.

		Expression  = [ Sequence ] { "|" [ Sequence ] } .
		Sequence    = Operand { Operand } .
		Operand     = Literal | Field | Indentation | Group | Option | Repetition .

		Literal     = string .
		Field       = FieldName [ ":" RuleName ] .
		FieldName   = identifier | "^" | "*" .

		Indent      = ">>" Operand Expression "<<" .
		Group       = "(" Expression ")" .
		Option      = "[" Expression "]" .
		Repetition  = "{" Expression [ "/" Expression ] "}" .

	TODO complete this comment
*/
type Format map [string] expr;


// ----------------------------------------------------------------------------
// Error handling

// Error describes an individual error. The position Pos, if valid,
// indicates the format source position the error relates to. The
// error is specified with the Msg string.
// 
type Error struct {
	Pos token.Position;
	Msg string;
}


// Error implements the os.Error interface.
func (e *Error) String() string {
	pos := "";
	if e.Pos.IsValid() {
		pos = fmt.Sprintf("%d:%d: ", e.Pos.Line, e.Pos.Column);
	}
	return pos + e.Msg;
}


// Multiple parser errors are returned as an ErrorList.
type ErrorList []*Error


// ErrorList implements the SortInterface.
func (p ErrorList) Len() int  { return len(p); }
func (p ErrorList) Swap(i, j int)  { p[i], p[j] = p[j], p[i]; }
func (p ErrorList) Less(i, j int) bool  { return p[i].Pos.Offset < p[j].Pos.Offset; }


// ErrorList implements the os.Error interface.
func (p ErrorList) String() string {
	switch len(p) {
	case 0: return "unspecified error";
	case 1: return p[0].String();
	}
	return fmt.Sprintf("%s (and %d more errors)", p[0].String(), len(p) - 1);
}


// ----------------------------------------------------------------------------
// Parsing

type parser struct {
	errors vector.Vector;
	scanner scanner.Scanner;
	pos token.Position;  // token position
	tok token.Token;  // one token look-ahead
	lit []byte;  // token literal

	packs map [string] string;  // PackageName -> ImportPath
	rules Format;  // RuleName -> Expression
}


// The parser implements scanner.Error.
func (p *parser) Error(pos token.Position, msg string) {
	// Don't collect errors that are on the same line as the previous error
	// in the hope to reduce the number of spurious errors due to incorrect
	// parser synchronization.
	if p.errors.Len() == 0 || p.errors.Last().(*Error).Pos.Line != pos.Line {
		p.errors.Push(&Error{pos, msg});
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


func (p *parser) parseTypeName() (string, bool) {
	pos := p.pos;
	name, is_ident := p.parseIdentifier(), true;
	if p.tok == token.PERIOD {
		// got a package name, lookup package
		if import_path, found := p.packs[name]; found {
			name = import_path;
		} else {
			p.Error(pos, "package not declared: " + name);
		}
		p.next();
		name, is_ident = name + "." + p.parseIdentifier(), false;
	}
	return name, is_ident;
}


// Parses a rule name and returns it. If the rule name is
// a package-qualified type name, the package name is resolved.
// The 2nd result value is true iff the rule name consists of a
// single identifier only (and thus could be a package name).
//
func (p *parser) parseRuleName() (string, bool) {
	name, is_ident := "", false;
	switch p.tok {
	case token.IDENT:
		name, is_ident = p.parseTypeName();
	case token.DEFAULT:
		name = "default";
		p.next();
	case token.QUO:
		name = "/";
		p.next();
	default:
		p.error_expected(p.pos, "rule name");
		p.next();  // make progress in any case
	}
	return name, is_ident;
}


func asLiteral(x interface{}) expr {
	s := x.(string);
	if len(s) > 0 && s[0] == '%' {
		// literals containing format characters are represented as strings
		return s;
	}
	// all other literals are represented as []byte for faster writing
	return io.StringBytes(s);
}


func (p *parser) parseLiteral() expr {
	if p.tok != token.STRING {
		p.expect(token.STRING);
		return "";
	}

	s, err := strconv.Unquote(string(p.lit));
	if err != nil {
		panic("scanner error");
	}
	p.next();

	// A string literal may contain newline characters and %-format specifiers.
	// To simplify and speed up printing of the literal, split it into segments
	// that start with "\n" or "%" (but noy "%%"), possibly followed by a last
	// segment that starts with some other character. If there is more than one
	// such segment, return a sequence of "simple" literals, otherwise just
	// return the string.

	// split string
	var list vector.Vector;
	list.Init(0);
	i0 := 0;
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\n':
			// next segment starts with '\n'
		case '%':
			if i+1 >= len(s) || s[i+1] == '%' {
				i++;
				continue;  //  "%%" is not a format-%
			}
			// next segment starts with '%'
		default:
			// all other cases do not split the string
			continue;
		}
		// split off the current segment
		if i0 < i {
			list.Push(s[i0 : i]);
			i0 = i;
		}
	}
	// the final segment may start with any character
	// (it is empty iff the string is empty)
	list.Push(s[i0 : len(s)]);

	// no need for a sequence there is only one segment
	if list.Len() == 1 {
		return asLiteral(list.At(0));
	}

	// convert list into a sequence
	seq := make(sequence, list.Len());
	for i := 0; i < list.Len(); i++ {
		seq[i] = asLiteral(list.At(i));
	}
	return seq;
}


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
		// TODO(gri) could use reflect.ExpandType() to lookup a field
		// at parse-time - would provide "compile-time" errors and
		// faster printing.
		fname = p.parseIdentifier();
	default:
		return nil;
	}

	var rule_name string;
	if p.tok == token.COLON {
		p.next();
		var _ bool;
		rule_name, _ = p.parseRuleName();
	}

	return &field{fname, rule_name};
}


func (p *parser) parseExpression() expr

func (p *parser) parseOperand() (x expr) {
	switch p.tok {
	case token.STRING:
		x = p.parseLiteral();

	case token.SHR:
		p.next();
		x = &indentation{p.parseOperand(), p.parseExpression()};
		p.expect(token.SHL);

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
		x = p.parseField();  // may be nil
	}

	return x;
}


func (p *parser) parseSequence() expr {
	var list vector.Vector;
	list.Init(0);

	for x := p.parseOperand(); x != nil; x = p.parseOperand() {
		list.Push(x);
	}

	// no need for a sequence if list.Len() < 2
	switch list.Len() {
	case 0: return nil;
	case 1: return list.At(0).(expr);
	}

	// convert list into a sequence
	seq := make(sequence, list.Len());
	for i := 0; i < list.Len(); i++ {
		seq[i] = list.At(i).(expr);
	}
	return seq;
}


func (p *parser) parseExpression() expr {
	var list vector.Vector;
	list.Init(0);

	for {
		x := p.parseSequence();
		if x != nil {
			list.Push(x);
		}
		if p.tok != token.OR {
			break;
		}
		p.next();
	}

	// no need for an alternatives if list.Len() < 2
	switch list.Len() {
	case 0: return nil;
	case 1: return list.At(0).(expr);
	}

	// convert list into a alternatives
	alt := make(alternatives, list.Len());
	for i := 0; i < list.Len(); i++ {
		alt[i] = list.At(i).(expr);
	}
	return alt;
}


func (p *parser) parseFormat() {
	for p.tok != token.EOF {
		pos := p.pos;

		name, is_ident := p.parseRuleName();
		switch p.tok {
		case token.STRING:
			// package declaration
			import_path, err := strconv.Unquote(string(p.lit));
			if err != nil {
				panic("scanner error");
			}
			p.next();

			// add package declaration
			if !is_ident {
				p.Error(pos, "illegal package name: " + name);
			} else if _, found := p.packs[name]; !found {
				p.packs[name] = import_path;
			} else {
				p.Error(pos, "package already declared: " + name);
			}

		case token.ASSIGN:
			// format rule
			p.next();
			x := p.parseExpression();

			// add rule
			if _, found := p.rules[name]; !found {
				p.rules[name] = x;
			} else {
				p.Error(pos, "format rule already declared: " + name);
			}

		default:
			p.error_expected(p.pos, "package declaration or format rule");
			p.next();  // make progress in any case
		}

		if p.tok == token.SEMICOLON {
			p.next();
		} else {
			break;
		}
	}
	p.expect(token.EOF);
}


func (p *parser) remap(pos token.Position, name string) string {
	i := strings.Index(name, ".");
	if i >= 0 {
		package_name := name[0 : i];
		type_name := name[i : len(name)];
		// lookup package
		if import_path, found := p.packs[package_name]; found {
			name = import_path + "." + type_name;
		} else {
			p.Error(pos, "package not declared: " + package_name);
		}
	}
	return name;
}


// Parse parses a set of format productions from source src. If there are no
// errors, the result is a Format and the error is nil. Otherwise the format
// is nil and a non-empty ErrorList is returned.
//
func Parse(src []byte, fmap FormatterMap) (Format, os.Error) {
	// parse source
	var p parser;
	p.errors.Init(0);
	p.scanner.Init(src, &p, false);
	p.next();
	p.packs = make(map [string] string);
	p.rules = make(Format);
	p.parseFormat();

	// add custom formatters, if any
	var invalidPos token.Position;
	for name, form := range fmap {
		name = p.remap(invalidPos, name);
		if t, found := p.rules[name]; !found {
			p.rules[name] = &custom{name, form};
		} else {
			var invalidPos token.Position;
			p.Error(invalidPos, "formatter already declared: " + name);
		}
	}

	// convert errors list, if any
	if p.errors.Len() > 0 {
		errors := make(ErrorList, p.errors.Len());
		for i := 0; i < p.errors.Len(); i++ {
			errors[i] = p.errors.At(i).(*Error);
		}
		return nil, errors;
	}

	return p.rules, nil;
}


// ----------------------------------------------------------------------------
// Formatting

// The current formatting state.
type state struct {
	f Format;  // the format used
	env interface{};  // the user-supplied environment, simply passed through
	def expr;  // the default rule, if any
	div expr;  // the global divider rule, if any
	writediv bool;  // true if the divider needs to be written
	errors chan os.Error;  // not chan *Error: errors <- nil would be wrong!
	indent io.ByteBuffer;  // the current indentation
}


func (ps *state) init(f Format, env interface{}, errors chan os.Error) {
	ps.f = f;
	ps.env = env;
	// if we have a default ("default") rule, cache it for fast access
	if def, has_def := f["default"]; has_def {
		ps.def = def;
	}
	// if we have a divider ("/") rule, cache it for fast access
	if div, has_div := f["/"]; has_div {
		ps.div = div;
	}
	ps.errors = errors;
}


func (ps *state) error(msg string) {
	ps.errors <- os.NewError(msg);
	runtime.Goexit();
}


// Get a field value given a field name. Returns the field value and
// the "embedding level" at which it was found. The embedding level
// is 0 for top-level fields in a struct.
//
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
	level := 1000;  // infinity (no struct has that many levels)
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
	if fexpr, found := ps.f[name]; found {
		return fexpr;
	}

	if ps.def != nil {
		return ps.def;
	}

	ps.error(fmt.Sprintf("no production for type: '%s'\n", name));
	return nil;
}


func (ps *state) printf(w io.Writer, fexpr expr, value reflect.Value, index int) bool


func (ps *state) printDiv(w io.Writer, value reflect.Value) {
	if ps.div != nil && ps.writediv {
		div := ps.div;
		ps.div = nil;
		ps.printf(w, div, value, 0);
		ps.div = div;
	}
	ps.writediv = true;
}


func (ps *state) writeIndented(w io.Writer, s []byte) {
	// write indent after each '\n'
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


// TODO complete this comment
// Returns true if a non-empty field value was found.
func (ps *state) printf(w io.Writer, fexpr expr, value reflect.Value, index int) bool {
	if fexpr == nil {
		return true;
	}

	switch t := fexpr.(type) {
	case alternatives:
		// - write first non-empty alternative
		// - result is not empty iff there is an non-empty alternative
		for _, x := range t {
			var buf io.ByteBuffer;
			if ps.printf(&buf, x, value, 0) {
				w.Write(buf.Data());
				return true;
			}
		}
		return false;

	case sequence:
		// - write every element of the sequence
		// - result is not empty iff no element was empty
		b := true;
		for _, x := range t {
			b = ps.printf(w, x, value, index) && b;
		}
		return b;

	case []byte:
		// write literal, may start with "\n"
		ps.printDiv(w, value);
		if len(t) > 0 && t[0] == '\n' && ps.indent.Len() > 0 {
			// newline must be followed by indentation
			w.Write([]byte{'\n'});
			w.Write(ps.indent.Data());
			t = t[1 : len(t)];
		}
		w.Write(t);
		return true;
		
	case string:
		// write format literal with value, starts with "%" (but not "%%")
		ps.printDiv(w, value);
		fmt.Fprintf(w, t, value.Interface());
		return true;

	case *field:
		// - write the contents of the field
		// - format is either the field format or the type-specific format
		// - result is not empty iff the field is not empty
		switch t.field_name {
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
				ps.error(fmt.Sprintf("error: * does not apply to `%s`\n", value.Type().Name()));
			}

		default:
			// field
			field, _ := getField(value, t.field_name);
			if field == nil {
				ps.error(fmt.Sprintf("error: no field `%s` in `%s`\n", t.field_name, value.Type().Name()));
			}
			value = field;
		}

		// field-specific rule name
		rule_name := t.rule_name;
		if rule_name == "" {
			rule_name = typename(value)
		}
		fexpr = ps.getFormat(rule_name);

		return ps.printf(w, fexpr, value, index);

	case *indentation:
		// - write the body within the given indentation
		// - the result is not empty iff the body is not empty
		saved_len := ps.indent.Len();
		ps.printf(&ps.indent, t.indent, value, index);  // add additional indentation
		b := ps.printf(w, t.body, value, index);
		ps.indent.Truncate(saved_len);  // reset indentation
		return b;

	case *option:
		// - write body if it is not empty
		// - the result is always not empty
		var buf io.ByteBuffer;
		if ps.printf(&buf, t.body, value, 0) {
			w.Write(buf.Data());
		}
		return true;

	case *repetition:
		// - write body until as long as it is not empty
		// - the result is always not empty
		var buf io.ByteBuffer;
		for i := 0; ps.printf(&buf, t.body, value, i); i++ {
			if i > 0 {
				ps.printf(w, t.div, value, i);
			}
			w.Write(buf.Data());
			buf.Reset();
		}
		return true;

	case *custom:
		// - invoke custom formatter
		var buf io.ByteBuffer;
		if t.form(&buf, ps.env, value.Interface(), t.rule_name) {
			ps.writeIndented(w, buf.Data());
			return true;
		}
		return false;
	}

	panic("unreachable");
	return false;
}


// Sandbox to wrap a writer.
// Counts total number of bytes written and handles write errors.
//
type sandbox struct {
	writer io.Writer;
	written int;
	errors chan os.Error;
}


// Write data to the sandboxed writer. If an error occurs, Write
// doesn't return. Instead it reports the error to the errors
// channel and exits the current goroutine.
//
func (s *sandbox) Write(data []byte) (int, os.Error) {
	n, err := s.writer.Write(data);
	s.written += n;
	if err != nil {
		s.errors <- err;
		runtime.Goexit();
	}
	return n, nil;
}


// Fprint formats each argument according to the format f
// and writes to w. The result is the total number of bytes
// written and an os.Error, if any.
//
func (f Format) Fprint(w io.Writer, env interface{}, args ...) (int, os.Error) {
	errors := make(chan os.Error);
	sw := sandbox{w, 0, errors};

	var ps state;
	ps.init(f, env, errors);

	go func() {
		value := reflect.NewValue(args).(reflect.StructValue);
		for i := 0; i < value.Len(); i++ {
			fld := value.Field(i);
			ps.printf(&sw, ps.getFormat(typename(fld)), fld, 0);
		}
		errors <- nil;  // no errors
	}();

	return sw.written, <-errors;
}


// Print formats each argument according to the format f
// and writes to standard output. The result is the total
// number of bytes written and an os.Error, if any.
//
func (f Format) Print(args ...) (int, os.Error) {
	return f.Fprint(os.Stdout, nil, args);
}


// Sprint formats each argument according to the format f
// and returns the resulting string. If an error occurs
// during formatting, the result contains the respective
// error message at the end.
//
func (f Format) Sprint(args ...) string {
	var buf io.ByteBuffer;
	n, err := f.Fprint(&buf, nil, args);
	if err != nil {
		fmt.Fprintf(&buf, "--- Sprint(%v) failed: %v", args, err);
	}
	return string(buf.Data());
}
