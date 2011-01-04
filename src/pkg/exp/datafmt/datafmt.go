// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*	The datafmt package implements syntax-directed, type-driven formatting
	of arbitrary data structures. Formatting a data structure consists of
	two phases: first, a parser reads a format specification and builds a
	"compiled" format. Then, the format can be applied repeatedly to
	arbitrary values. Applying a format to a value evaluates to a []byte
	containing the formatted value bytes, or nil.

	A format specification is a set of package declarations and format rules:

		Format      = [ Entry { ";" Entry } [ ";" ] ] .
		Entry       = PackageDecl | FormatRule .

	(The syntax of a format specification is presented in the same EBNF
	notation as used in the Go language specification. The syntax of white
	space, comments, identifiers, and string literals is the same as in Go.)

	A package declaration binds a package name (such as 'ast') to a
	package import path (such as '"go/ast"'). Each package used (in
	a type name, see below) must be declared once before use.

		PackageDecl = PackageName ImportPath .
		PackageName = identifier .
		ImportPath  = string .

	A format rule binds a rule name to a format expression. A rule name
	may be a type name or one of the special names 'default' or '/'.
	A type name may be the name of a predeclared type (for example, 'int',
	'float32', etc.), the package-qualified name of a user-defined type
	(for example, 'ast.MapType'), or an identifier indicating the structure
	of unnamed composite types ('array', 'chan', 'func', 'interface', 'map',
	or 'ptr'). Each rule must have a unique name; rules can be declared in
	any order.

		FormatRule  = RuleName "=" Expression .
		RuleName    = TypeName | "default" | "/" .
		TypeName    = [ PackageName "." ] identifier .

	To format a value, the value's type name is used to select the format rule
	(there is an override mechanism, see below). The format expression of the
	selected rule specifies how the value is formatted. Each format expression,
	when applied to a value, evaluates to a byte sequence or nil.

	In its most general form, a format expression is a list of alternatives,
	each of which is a sequence of operands:

		Expression  = [ Sequence ] { "|" [ Sequence ] } .
		Sequence    = Operand { Operand } .

	The formatted result produced by an expression is the result of the first
	alternative sequence that evaluates to a non-nil result; if there is no
	such alternative, the expression evaluates to nil. The result produced by
	an operand sequence is the concatenation of the results of its operands.
	If any operand in the sequence evaluates to nil, the entire sequence
	evaluates to nil.

	There are five kinds of operands:

		Operand     = Literal | Field | Group | Option | Repetition .

	Literals evaluate to themselves, with two substitutions. First,
	%-formats expand in the manner of fmt.Printf, with the current value
	passed as the parameter. Second, the current indentation (see below)
	is inserted after every newline or form feed character.

		Literal     = string .

	This table shows string literals applied to the value 42 and the
	corresponding formatted result:

		"foo"       foo
		"%x"        2a
		"x = %d"    x = 42
		"%#x = %d"  0x2a = 42

	A field operand is a field name optionally followed by an alternate
	rule name. The field name may be an identifier or one of the special
	names @ or *.

		Field       = FieldName [ ":" RuleName ] .
		FieldName   = identifier | "@" | "*" .

	If the field name is an identifier, the current value must be a struct,
	and there must be a field with that name in the struct. The same lookup
	rules apply as in the Go language (for instance, the name of an anonymous
	field is the unqualified type name). The field name denotes the field
	value in the struct. If the field is not found, formatting is aborted
	and an error message is returned. (TODO consider changing the semantics
	such that if a field is not found, it evaluates to nil).

	The special name '@' denotes the current value.

	The meaning of the special name '*' depends on the type of the current
	value:

		array, slice types   array, slice element (inside {} only, see below)
		interfaces           value stored in interface
		pointers             value pointed to by pointer

	(Implementation restriction: channel, function and map types are not
	supported due to missing reflection support).

	Fields are evaluated as follows: If the field value is nil, or an array
	or slice element does not exist, the result is nil (see below for details
	on array/slice elements). If the value is not nil the field value is
	formatted (recursively) using the rule corresponding to its type name,
	or the alternate rule name, if given.

	The following example shows a complete format specification for a
	struct 'myPackage.Point'. Assume the package

		package myPackage  // in directory myDir/myPackage
		type Point struct {
			name string;
			x, y int;
		}

	Applying the format specification

		myPackage "myDir/myPackage";
		int = "%d";
		hexInt = "0x%x";
		string = "---%s---";
		myPackage.Point = name "{" x ", " y:hexInt "}";

	to the value myPackage.Point{"foo", 3, 15} results in

		---foo---{3, 0xf}

	Finally, an operand may be a grouped, optional, or repeated expression.
	A grouped expression ("group") groups a more complex expression (body)
	so that it can be used in place of a single operand:

		Group       = "(" [ Indentation ">>" ] Body ")" .
		Indentation = Expression .
		Body        = Expression .

	A group body may be prefixed by an indentation expression followed by '>>'.
	The indentation expression is applied to the current value like any other
	expression and the result, if not nil, is appended to the current indentation
	during the evaluation of the body (see also formatting state, below).

	An optional expression ("option") is enclosed in '[]' brackets.

		Option      = "[" Body "]" .

	An option evaluates to its body, except that if the body evaluates to nil,
	the option expression evaluates to an empty []byte. Thus an option's purpose
	is to protect the expression containing the option from a nil operand.

	A repeated expression ("repetition") is enclosed in '{}' braces.

		Repetition  = "{" Body [ "/" Separator ] "}" .
		Separator   = Expression .

	A repeated expression is evaluated as follows: The body is evaluated
	repeatedly and its results are concatenated until the body evaluates
	to nil. The result of the repetition is the (possibly empty) concatenation,
	but it is never nil. An implicit index is supplied for the evaluation of
	the body: that index is used to address elements of arrays or slices. If
	the corresponding elements do not exist, the field denoting the element
	evaluates to nil (which in turn may terminate the repetition).

	The body of a repetition may be followed by a '/' and a "separator"
	expression. If the separator is present, it is invoked between repetitions
	of the body.

	The following example shows a complete format specification for formatting
	a slice of unnamed type. Applying the specification

		int = "%b";
		array = { * / ", " };  // array is the type name for an unnamed slice

	to the value '[]int{2, 3, 5, 7}' results in

		10, 11, 101, 111

	Default rule: If a format rule named 'default' is present, it is used for
	formatting a value if no other rule was found. A common default rule is

		default = "%v"

	to provide default formatting for basic types without having to specify
	a specific rule for each basic type.

	Global separator rule: If a format rule named '/' is present, it is
	invoked with the current value between literals. If the separator
	expression evaluates to nil, it is ignored.

	For instance, a global separator rule may be used to punctuate a sequence
	of values with commas. The rules:

		default = "%v";
		/ = ", ";

	will format an argument list by printing each one in its default format,
	separated by a comma and a space.
*/
package datafmt

import (
	"bytes"
	"fmt"
	"go/token"
	"io"
	"os"
	"reflect"
	"runtime"
)


// ----------------------------------------------------------------------------
// Format representation

// Custom formatters implement the Formatter function type.
// A formatter is invoked with the current formatting state, the
// value to format, and the rule name under which the formatter
// was installed (the same formatter function may be installed
// under different names). The formatter may access the current state
// to guide formatting and use State.Write to append to the state's
// output.
//
// A formatter must return a boolean value indicating if it evaluated
// to a non-nil value (true), or a nil value (false).
//
type Formatter func(state *State, value interface{}, ruleName string) bool


// A FormatterMap is a set of custom formatters.
// It maps a rule name to a formatter function.
//
type FormatterMap map[string]Formatter


// A parsed format expression is built from the following nodes.
//
type (
	expr interface{}

	alternatives []expr // x | y | z

	sequence []expr // x y z

	literal [][]byte // a list of string segments, possibly starting with '%'

	field struct {
		fieldName string // including "@", "*"
		ruleName  string // "" if no rule name specified
	}

	group struct {
		indent, body expr // (indent >> body)
	}

	option struct {
		body expr // [body]
	}

	repetition struct {
		body, separator expr // {body / separator}
	}

	custom struct {
		ruleName string
		fun      Formatter
	}
)


// A Format is the result of parsing a format specification.
// The format may be applied repeatedly to format values.
//
type Format map[string]expr


// ----------------------------------------------------------------------------
// Formatting

// An application-specific environment may be provided to Format.Apply;
// the environment is available inside custom formatters via State.Env().
// Environments must implement copying; the Copy method must return an
// complete copy of the receiver. This is necessary so that the formatter
// can save and restore an environment (in case of an absent expression).
//
// If the Environment doesn't change during formatting (this is under
// control of the custom formatters), the Copy function can simply return
// the receiver, and thus can be very light-weight.
//
type Environment interface {
	Copy() Environment
}


// State represents the current formatting state.
// It is provided as argument to custom formatters.
//
type State struct {
	fmt       Format         // format in use
	env       Environment    // user-supplied environment
	errors    chan os.Error  // not chan *Error (errors <- nil would be wrong!)
	hasOutput bool           // true after the first literal has been written
	indent    bytes.Buffer   // current indentation
	output    bytes.Buffer   // format output
	linePos   token.Position // position of line beginning (Column == 0)
	default_  expr           // possibly nil
	separator expr           // possibly nil
}


func newState(fmt Format, env Environment, errors chan os.Error) *State {
	s := new(State)
	s.fmt = fmt
	s.env = env
	s.errors = errors
	s.linePos = token.Position{Line: 1}

	// if we have a default rule, cache it's expression for fast access
	if x, found := fmt["default"]; found {
		s.default_ = x
	}

	// if we have a global separator rule, cache it's expression for fast access
	if x, found := fmt["/"]; found {
		s.separator = x
	}

	return s
}


// Env returns the environment passed to Format.Apply.
func (s *State) Env() interface{} { return s.env }


// LinePos returns the position of the current line beginning
// in the state's output buffer. Line numbers start at 1.
//
func (s *State) LinePos() token.Position { return s.linePos }


// Pos returns the position of the next byte to be written to the
// output buffer. Line numbers start at 1.
//
func (s *State) Pos() token.Position {
	offs := s.output.Len()
	return token.Position{Line: s.linePos.Line, Column: offs - s.linePos.Offset, Offset: offs}
}


// Write writes data to the output buffer, inserting the indentation
// string after each newline or form feed character. It cannot return an error.
//
func (s *State) Write(data []byte) (int, os.Error) {
	n := 0
	i0 := 0
	for i, ch := range data {
		if ch == '\n' || ch == '\f' {
			// write text segment and indentation
			n1, _ := s.output.Write(data[i0 : i+1])
			n2, _ := s.output.Write(s.indent.Bytes())
			n += n1 + n2
			i0 = i + 1
			s.linePos.Offset = s.output.Len()
			s.linePos.Line++
		}
	}
	n3, _ := s.output.Write(data[i0:])
	return n + n3, nil
}


type checkpoint struct {
	env       Environment
	hasOutput bool
	outputLen int
	linePos   token.Position
}


func (s *State) save() checkpoint {
	saved := checkpoint{nil, s.hasOutput, s.output.Len(), s.linePos}
	if s.env != nil {
		saved.env = s.env.Copy()
	}
	return saved
}


func (s *State) restore(m checkpoint) {
	s.env = m.env
	s.output.Truncate(m.outputLen)
}


func (s *State) error(msg string) {
	s.errors <- os.NewError(msg)
	runtime.Goexit()
}


// TODO At the moment, unnamed types are simply mapped to the default
//      names below. For instance, all unnamed arrays are mapped to
//      'array' which is not really sufficient. Eventually one may want
//      to be able to specify rules for say an unnamed slice of T.
//

func typename(typ reflect.Type) string {
	switch typ.(type) {
	case *reflect.ArrayType:
		return "array"
	case *reflect.SliceType:
		return "array"
	case *reflect.ChanType:
		return "chan"
	case *reflect.FuncType:
		return "func"
	case *reflect.InterfaceType:
		return "interface"
	case *reflect.MapType:
		return "map"
	case *reflect.PtrType:
		return "ptr"
	}
	return typ.String()
}

func (s *State) getFormat(name string) expr {
	if fexpr, found := s.fmt[name]; found {
		return fexpr
	}

	if s.default_ != nil {
		return s.default_
	}

	s.error(fmt.Sprintf("no format rule for type: '%s'", name))
	return nil
}


// eval applies a format expression fexpr to a value. If the expression
// evaluates internally to a non-nil []byte, that slice is appended to
// the state's output buffer and eval returns true. Otherwise, eval
// returns false and the state remains unchanged.
//
func (s *State) eval(fexpr expr, value reflect.Value, index int) bool {
	// an empty format expression always evaluates
	// to a non-nil (but empty) []byte
	if fexpr == nil {
		return true
	}

	switch t := fexpr.(type) {
	case alternatives:
		// append the result of the first alternative that evaluates to
		// a non-nil []byte to the state's output
		mark := s.save()
		for _, x := range t {
			if s.eval(x, value, index) {
				return true
			}
			s.restore(mark)
		}
		return false

	case sequence:
		// append the result of all operands to the state's output
		// unless a nil result is encountered
		mark := s.save()
		for _, x := range t {
			if !s.eval(x, value, index) {
				s.restore(mark)
				return false
			}
		}
		return true

	case literal:
		// write separator, if any
		if s.hasOutput {
			// not the first literal
			if s.separator != nil {
				sep := s.separator // save current separator
				s.separator = nil  // and disable it (avoid recursion)
				mark := s.save()
				if !s.eval(sep, value, index) {
					s.restore(mark)
				}
				s.separator = sep // enable it again
			}
		}
		s.hasOutput = true
		// write literal segments
		for _, lit := range t {
			if len(lit) > 1 && lit[0] == '%' {
				// segment contains a %-format at the beginning
				if lit[1] == '%' {
					// "%%" is printed as a single "%"
					s.Write(lit[1:])
				} else {
					// use s instead of s.output to get indentation right
					fmt.Fprintf(s, string(lit), value.Interface())
				}
			} else {
				// segment contains no %-formats
				s.Write(lit)
			}
		}
		return true // a literal never evaluates to nil

	case *field:
		// determine field value
		switch t.fieldName {
		case "@":
			// field value is current value

		case "*":
			// indirection: operation is type-specific
			switch v := value.(type) {
			case *reflect.ArrayValue:
				if v.Len() <= index {
					return false
				}
				value = v.Elem(index)

			case *reflect.SliceValue:
				if v.IsNil() || v.Len() <= index {
					return false
				}
				value = v.Elem(index)

			case *reflect.MapValue:
				s.error("reflection support for maps incomplete")

			case *reflect.PtrValue:
				if v.IsNil() {
					return false
				}
				value = v.Elem()

			case *reflect.InterfaceValue:
				if v.IsNil() {
					return false
				}
				value = v.Elem()

			case *reflect.ChanValue:
				s.error("reflection support for chans incomplete")

			case *reflect.FuncValue:
				s.error("reflection support for funcs incomplete")

			default:
				s.error(fmt.Sprintf("error: * does not apply to `%s`", value.Type()))
			}

		default:
			// value is value of named field
			var field reflect.Value
			if sval, ok := value.(*reflect.StructValue); ok {
				field = sval.FieldByName(t.fieldName)
				if field == nil {
					// TODO consider just returning false in this case
					s.error(fmt.Sprintf("error: no field `%s` in `%s`", t.fieldName, value.Type()))
				}
			}
			value = field
		}

		// determine rule
		ruleName := t.ruleName
		if ruleName == "" {
			// no alternate rule name, value type determines rule
			ruleName = typename(value.Type())
		}
		fexpr = s.getFormat(ruleName)

		mark := s.save()
		if !s.eval(fexpr, value, index) {
			s.restore(mark)
			return false
		}
		return true

	case *group:
		// remember current indentation
		indentLen := s.indent.Len()

		// update current indentation
		mark := s.save()
		s.eval(t.indent, value, index)
		// if the indentation evaluates to nil, the state's output buffer
		// didn't change - either way it's ok to append the difference to
		// the current identation
		s.indent.Write(s.output.Bytes()[mark.outputLen:s.output.Len()])
		s.restore(mark)

		// format group body
		mark = s.save()
		b := true
		if !s.eval(t.body, value, index) {
			s.restore(mark)
			b = false
		}

		// reset indentation
		s.indent.Truncate(indentLen)
		return b

	case *option:
		// evaluate the body and append the result to the state's output
		// buffer unless the result is nil
		mark := s.save()
		if !s.eval(t.body, value, 0) { // TODO is 0 index correct?
			s.restore(mark)
		}
		return true // an option never evaluates to nil

	case *repetition:
		// evaluate the body and append the result to the state's output
		// buffer until a result is nil
		for i := 0; ; i++ {
			mark := s.save()
			// write separator, if any
			if i > 0 && t.separator != nil {
				// nil result from separator is ignored
				mark := s.save()
				if !s.eval(t.separator, value, i) {
					s.restore(mark)
				}
			}
			if !s.eval(t.body, value, i) {
				s.restore(mark)
				break
			}
		}
		return true // a repetition never evaluates to nil

	case *custom:
		// invoke the custom formatter to obtain the result
		mark := s.save()
		if !t.fun(s, value.Interface(), t.ruleName) {
			s.restore(mark)
			return false
		}
		return true
	}

	panic("unreachable")
	return false
}


// Eval formats each argument according to the format
// f and returns the resulting []byte and os.Error. If
// an error occurred, the []byte contains the partially
// formatted result. An environment env may be passed
// in which is available in custom formatters through
// the state parameter.
//
func (f Format) Eval(env Environment, args ...interface{}) ([]byte, os.Error) {
	if f == nil {
		return nil, os.NewError("format is nil")
	}

	errors := make(chan os.Error)
	s := newState(f, env, errors)

	go func() {
		for _, v := range args {
			fld := reflect.NewValue(v)
			if fld == nil {
				errors <- os.NewError("nil argument")
				return
			}
			mark := s.save()
			if !s.eval(s.getFormat(typename(fld.Type())), fld, 0) { // TODO is 0 index correct?
				s.restore(mark)
			}
		}
		errors <- nil // no errors
	}()

	err := <-errors
	return s.output.Bytes(), err
}


// ----------------------------------------------------------------------------
// Convenience functions

// Fprint formats each argument according to the format f
// and writes to w. The result is the total number of bytes
// written and an os.Error, if any.
//
func (f Format) Fprint(w io.Writer, env Environment, args ...interface{}) (int, os.Error) {
	data, err := f.Eval(env, args...)
	if err != nil {
		// TODO should we print partial result in case of error?
		return 0, err
	}
	return w.Write(data)
}


// Print formats each argument according to the format f
// and writes to standard output. The result is the total
// number of bytes written and an os.Error, if any.
//
func (f Format) Print(args ...interface{}) (int, os.Error) {
	return f.Fprint(os.Stdout, nil, args...)
}


// Sprint formats each argument according to the format f
// and returns the resulting string. If an error occurs
// during formatting, the result string contains the
// partially formatted result followed by an error message.
//
func (f Format) Sprint(args ...interface{}) string {
	var buf bytes.Buffer
	_, err := f.Fprint(&buf, nil, args...)
	if err != nil {
		var i interface{} = args
		fmt.Fprintf(&buf, "--- Sprint(%s) failed: %v", fmt.Sprint(i), err)
	}
	return buf.String()
}
