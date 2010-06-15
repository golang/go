// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	Package fmt implements formatted I/O with functions analogous
	to C's printf and scanf.  The format 'verbs' are derived from C's but
	are simpler.

	Printing:

	The verbs:

	General:
		%v	the value in a default format.
			when printing structs, the plus flag (%+v) adds field names
		%#v	a Go-syntax representation of the value
		%T	a Go-syntax representation of the type of the value

	Boolean:
		%t	the word true or false
	Integer:
		%b	base 2
		%c	the character represented by the corresponding Unicode code point
		%d	base 10
		%o	base 8
		%x	base 16, with lower-case letters for a-f
		%X	base 16, with upper-case letters for A-F
	Floating-point and complex constituents:
		%e	scientific notation, e.g. -1234.456e+78
		%E	scientific notation, e.g. -1234.456E+78
		%f	decimal point but no exponent, e.g. 123.456
		%g	whichever of %e or %f produces more compact output
		%G	whichever of %E or %f produces more compact output
	String and slice of bytes:
		%s	the uninterpreted bytes of the string or slice
		%q	a double-quoted string safely escaped with Go syntax
		%x	base 16 notation with two characters per byte
	Pointer:
		%p	base 16 notation, with leading 0x

	There is no 'u' flag.  Integers are printed unsigned if they have unsigned type.
	Similarly, there is no need to specify the size of the operand (int8, int64).

	For numeric values, the width and precision flags control
	formatting; width sets the width of the field, precision the
	number of places after the decimal, if appropriate.  The
	format %6.2f prints 123.45. The width of a field is the number
	of Unicode code points in the string. This differs from C's printf where
	the field width is the number of bytes.

	Other flags:
		+	always print a sign for numeric values
		-	pad with spaces on the right rather than the left (left-justify the field)
		#	alternate format: add leading 0 for octal (%#o), 0x for hex (%#x);
			suppress 0x for %p (%#p);
			print a raw (backquoted) string if possible for %q (%#q)
		' '	(space) leave a space for elided sign in numbers (% d);
			put spaces between bytes printing strings or slices in hex (% x)
		0	pad with leading zeros rather than spaces

	For each Printf-like function, there is also a Print function
	that takes no format and is equivalent to saying %v for every
	operand.  Another variant Println inserts blanks between
	operands and appends a newline.

	Regardless of the verb, if an operand is an interface value,
	the internal concrete value is used, not the interface itself.
	Thus:
		var i interface{} = 23;
		fmt.Printf("%v\n", i);
	will print 23.

	If an operand implements interface Formatter, that interface
	can be used for fine control of formatting.

	If an operand implements method String() string that method
	will be used to conver the object to a string, which will then
	be formatted as required by the verb (if any). To avoid
	recursion in cases such as
		type X int
		func (x X) String() string { return Sprintf("%d", x) }
	cast the value before recurring:
		func (x X) String() string { return Sprintf("%d", int(x)) }

	Scanning:

	An analogous set of functions scans formatted text to yield
	values.  Scan, Scanf and Scanln read from os.Stdin; Fscan,
	Fscanf and Fscanln read from a specified os.Reader; Sscan,
	Sscanf and Sscanln read from an argument string.  Sscanln,
	Fscanln and Sscanln stop scanning at a newline and require that
	the items be followed by one; the other routines treat newlines
	as spaces.

	Scanf, Fscanf, and Sscanf parse the arguments according to a
	format string, analogous to that of Printf.  For example, "%x"
	will scan an integer as a hexadecimal number, and %v will scan
	the default representation format for the value.

	The formats behave analogously to those of Printf with the
	following exceptions:

	%p is not implemented
	%T is not implemented
	%e %E %f %F %g %g are all equivalent and scan any floating
		point or complex value
	%s and %v on strings scan a space-delimited token

	Width is interpreted in the input text (%5s means at most
	five runes of input will be read to scan a string) but there
	is no syntax for scanning with a precision (no %5.2f, just
	%5f).

	When scanning with a format, all non-empty runs of space
	characters (including newline) are equivalent to a single
	space in both the format and the input.  With that proviso,
	text in the format string must match the input text; scanning
	stops if it does not, with the return value of the function
	indicating the number of arguments scanned.

	In all the scanning functions, if an operand implements method
	Scan (that is, it implements the Scanner interface) that
	method will be used to scan the text for that operand.  Also,
	if the number of arguments scanned is less than the number of
	arguments provided, an error is returned.

	All arguments to be scanned must be either pointers to basic
	types or implementations of the Scanner interface.
*/
package fmt

import (
	"bytes"
	"io"
	"os"
	"reflect"
	"utf8"
)

// Some constants in the form of bytes, to avoid string overhead.
// Needlessly fastidious, I suppose.
var (
	commaSpaceBytes = []byte(", ")
	nilAngleBytes   = []byte("<nil>")
	nilParenBytes   = []byte("(nil)")
	nilBytes        = []byte("nil")
	mapBytes        = []byte("map[")
	missingBytes    = []byte("missing")
	extraBytes      = []byte("?(extra ")
	irparenBytes    = []byte("i)")
	bytesBytes      = []byte("[]byte{")
)

// State represents the printer state passed to custom formatters.
// It provides access to the io.Writer interface plus information about
// the flags and options for the operand's format specifier.
type State interface {
	// Write is the function to call to emit formatted output to be printed.
	Write(b []byte) (ret int, err os.Error)
	// Width returns the value of the width option and whether it has been set.
	Width() (wid int, ok bool)
	// Precision returns the value of the precision option and whether it has been set.
	Precision() (prec int, ok bool)

	// Flag returns whether the flag c, a character, has been set.
	Flag(int) bool
}

// Formatter is the interface implemented by values with a custom formatter.
// The implementation of Format may call Sprintf or Fprintf(f) etc.
// to generate its output.
type Formatter interface {
	Format(f State, c int)
}

// Stringer is implemented by any value that has a String method(),
// which defines the ``native'' format for that value.
// The String method is used to print values passed as an operand
// to a %s or %v format or to an unformatted printer such as Print.
type Stringer interface {
	String() string
}

// GoStringer is implemented by any value that has a GoString() method,
// which defines the Go syntax for that value.
// The GoString method is used to print values passed as an operand
// to a %#v format.
type GoStringer interface {
	GoString() string
}

type pp struct {
	n       int
	buf     bytes.Buffer
	runeBuf [utf8.UTFMax]byte
	fmt     fmt
}

// A leaky bucket of reusable pp structures.
var ppFree = make(chan *pp, 100)

// Allocate a new pp struct.  Probably can grab the previous one from ppFree.
func newPrinter() *pp {
	p, ok := <-ppFree
	if !ok {
		p = new(pp)
	}
	p.fmt.init(&p.buf)
	return p
}

// Save used pp structs in ppFree; avoids an allocation per invocation.
func (p *pp) free() {
	// Don't hold on to pp structs with large buffers.
	if cap(p.buf.Bytes()) > 1024 {
		return
	}
	p.buf.Reset()
	_ = ppFree <- p
}

func (p *pp) Width() (wid int, ok bool) { return p.fmt.wid, p.fmt.widPresent }

func (p *pp) Precision() (prec int, ok bool) { return p.fmt.prec, p.fmt.precPresent }

func (p *pp) Flag(b int) bool {
	switch b {
	case '-':
		return p.fmt.minus
	case '+':
		return p.fmt.plus
	case '#':
		return p.fmt.sharp
	case ' ':
		return p.fmt.space
	case '0':
		return p.fmt.zero
	}
	return false
}

func (p *pp) add(c int) {
	if c < utf8.RuneSelf {
		p.buf.WriteByte(byte(c))
	} else {
		w := utf8.EncodeRune(c, p.runeBuf[0:])
		p.buf.Write(p.runeBuf[0:w])
	}
}

// Implement Write so we can call Fprintf on a pp (through State), for
// recursive use in custom verbs.
func (p *pp) Write(b []byte) (ret int, err os.Error) {
	return p.buf.Write(b)
}

// These routines end in 'f' and take a format string.

// Fprintf formats according to a format specifier and writes to w.
func Fprintf(w io.Writer, format string, a ...interface{}) (n int, error os.Error) {
	p := newPrinter()
	p.doPrintf(format, a)
	n64, error := p.buf.WriteTo(w)
	p.free()
	return int(n64), error
}

// Printf formats according to a format specifier and writes to standard output.
func Printf(format string, a ...interface{}) (n int, errno os.Error) {
	n, errno = Fprintf(os.Stdout, format, a)
	return n, errno
}

// Sprintf formats according to a format specifier and returns the resulting string.
func Sprintf(format string, a ...interface{}) string {
	p := newPrinter()
	p.doPrintf(format, a)
	s := p.buf.String()
	p.free()
	return s
}

// These routines do not take a format string

// Fprint formats using the default formats for its operands and writes to w.
// Spaces are added between operands when neither is a string.
func Fprint(w io.Writer, a ...interface{}) (n int, error os.Error) {
	p := newPrinter()
	p.doPrint(a, false, false)
	n64, error := p.buf.WriteTo(w)
	p.free()
	return int(n64), error
}

// Print formats using the default formats for its operands and writes to standard output.
// Spaces are added between operands when neither is a string.
func Print(a ...interface{}) (n int, errno os.Error) {
	n, errno = Fprint(os.Stdout, a)
	return n, errno
}

// Sprint formats using the default formats for its operands and returns the resulting string.
// Spaces are added between operands when neither is a string.
func Sprint(a ...interface{}) string {
	p := newPrinter()
	p.doPrint(a, false, false)
	s := p.buf.String()
	p.free()
	return s
}

// These routines end in 'ln', do not take a format string,
// always add spaces between operands, and add a newline
// after the last operand.

// Fprintln formats using the default formats for its operands and writes to w.
// Spaces are always added between operands and a newline is appended.
func Fprintln(w io.Writer, a ...interface{}) (n int, error os.Error) {
	p := newPrinter()
	p.doPrint(a, true, true)
	n64, error := p.buf.WriteTo(w)
	p.free()
	return int(n64), error
}

// Println formats using the default formats for its operands and writes to standard output.
// Spaces are always added between operands and a newline is appended.
func Println(a ...interface{}) (n int, errno os.Error) {
	n, errno = Fprintln(os.Stdout, a)
	return n, errno
}

// Sprintln formats using the default formats for its operands and returns the resulting string.
// Spaces are always added between operands and a newline is appended.
func Sprintln(a ...interface{}) string {
	p := newPrinter()
	p.doPrint(a, true, true)
	s := p.buf.String()
	p.free()
	return s
}


// Get the i'th arg of the struct value.
// If the arg itself is an interface, return a value for
// the thing inside the interface, not the interface itself.
func getField(v *reflect.StructValue, i int) reflect.Value {
	val := v.Field(i)
	if i, ok := val.(*reflect.InterfaceValue); ok {
		if inter := i.Interface(); inter != nil {
			return reflect.NewValue(inter)
		}
	}
	return val
}

// Convert ASCII to integer.  n is 0 (and got is false) if no number present.
func parsenum(s string, start, end int) (num int, isnum bool, newi int) {
	if start >= end {
		return 0, false, end
	}
	for newi = start; newi < end && '0' <= s[newi] && s[newi] <= '9'; newi++ {
		num = num*10 + int(s[newi]-'0')
		isnum = true
	}
	return
}

// Reflection values like reflect.FuncValue implement this method. We use it for %p.
type uintptrGetter interface {
	Get() uintptr
}

func (p *pp) unknownType(v interface{}) {
	if v == nil {
		p.buf.Write(nilAngleBytes)
		return
	}
	p.buf.WriteByte('?')
	p.buf.WriteString(reflect.Typeof(v).String())
	p.buf.WriteByte('?')
}

func (p *pp) badVerb(verb int, val interface{}) {
	p.add('%')
	p.add(verb)
	p.add('(')
	if val == nil {
		p.buf.Write(nilAngleBytes)
	} else {
		p.buf.WriteString(reflect.Typeof(val).String())
		p.add('=')
		p.printField(val, 'v', false, false, 0)
	}
	p.add(')')
}

func (p *pp) fmtBool(v bool, verb int, value interface{}) {
	switch verb {
	case 't', 'v':
		p.fmt.fmt_boolean(v)
	default:
		p.badVerb(verb, value)
	}
}

// fmtC formats a rune for the 'c' format.
func (p *pp) fmtC(c int64) {
	rune := int(c) // Check for overflow.
	if int64(rune) != c {
		rune = utf8.RuneError
	}
	w := utf8.EncodeRune(rune, p.runeBuf[0:utf8.UTFMax])
	p.fmt.pad(p.runeBuf[0:w])
}

func (p *pp) fmtInt64(v int64, verb int, value interface{}) {
	switch verb {
	case 'b':
		p.fmt.integer(v, 2, signed, ldigits)
	case 'c':
		p.fmtC(v)
	case 'd', 'v':
		p.fmt.integer(v, 10, signed, ldigits)
	case 'o':
		p.fmt.integer(v, 8, signed, ldigits)
	case 'x':
		p.fmt.integer(v, 16, signed, ldigits)
	case 'X':
		p.fmt.integer(v, 16, signed, udigits)
	default:
		p.badVerb(verb, value)
	}
}

// fmt_sharpHex64 formats a uint64 in hexadecimal and prefixes it with 0x by
// temporarily turning on the sharp flag.
func (p *pp) fmt0x64(v uint64) {
	sharp := p.fmt.sharp
	p.fmt.sharp = true // turn on 0x
	p.fmt.integer(int64(v), 16, unsigned, ldigits)
	p.fmt.sharp = sharp
}

func (p *pp) fmtUint64(v uint64, verb int, sharp bool, value interface{}) {
	switch verb {
	case 'b':
		p.fmt.integer(int64(v), 2, unsigned, ldigits)
	case 'c':
		p.fmtC(int64(v))
	case 'd':
		p.fmt.integer(int64(v), 10, unsigned, ldigits)
	case 'v':
		if sharp {
			p.fmt0x64(v)
		} else {
			p.fmt.integer(int64(v), 10, unsigned, ldigits)
		}
	case 'o':
		p.fmt.integer(int64(v), 8, unsigned, ldigits)
	case 'x':
		p.fmt.integer(int64(v), 16, unsigned, ldigits)
	case 'X':
		p.fmt.integer(int64(v), 16, unsigned, udigits)
	default:
		p.badVerb(verb, value)
	}
}

var floatBits = reflect.Typeof(float(0)).Size() * 8

func (p *pp) fmtFloat32(v float32, verb int, value interface{}) {
	switch verb {
	case 'b':
		p.fmt.fmt_fb32(v)
	case 'e':
		p.fmt.fmt_e32(v)
	case 'E':
		p.fmt.fmt_E32(v)
	case 'f':
		p.fmt.fmt_f32(v)
	case 'g', 'v':
		p.fmt.fmt_g32(v)
	case 'G':
		p.fmt.fmt_G32(v)
	default:
		p.badVerb(verb, value)
	}
}

func (p *pp) fmtFloat64(v float64, verb int, value interface{}) {
	switch verb {
	case 'b':
		p.fmt.fmt_fb64(v)
	case 'e':
		p.fmt.fmt_e64(v)
	case 'E':
		p.fmt.fmt_E64(v)
	case 'f':
		p.fmt.fmt_f64(v)
	case 'g', 'v':
		p.fmt.fmt_g64(v)
	case 'G':
		p.fmt.fmt_G64(v)
	default:
		p.badVerb(verb, value)
	}
}

var complexBits = reflect.Typeof(complex(0i)).Size() * 8

func (p *pp) fmtComplex64(v complex64, verb int, value interface{}) {
	switch verb {
	case 'e', 'E', 'f', 'F', 'g', 'G':
		p.fmt.fmt_c64(v, verb)
	case 'v':
		p.fmt.fmt_c64(v, 'g')
	default:
		p.badVerb(verb, value)
	}
}

func (p *pp) fmtComplex128(v complex128, verb int, value interface{}) {
	switch verb {
	case 'e', 'E', 'f', 'F', 'g', 'G':
		p.fmt.fmt_c128(v, verb)
	case 'v':
		p.fmt.fmt_c128(v, 'g')
	default:
		p.badVerb(verb, value)
	}
}

func (p *pp) fmtString(v string, verb int, sharp bool, value interface{}) {
	switch verb {
	case 'v':
		if sharp {
			p.fmt.fmt_q(v)
		} else {
			p.fmt.fmt_s(v)
		}
	case 's':
		p.fmt.fmt_s(v)
	case 'x':
		p.fmt.fmt_sx(v)
	case 'X':
		p.fmt.fmt_sX(v)
	case 'q':
		p.fmt.fmt_q(v)
	default:
		p.badVerb(verb, value)
	}
}

func (p *pp) fmtBytes(v []byte, verb int, sharp bool, depth int, value interface{}) {
	if verb == 'v' {
		if p.fmt.sharp {
			p.buf.Write(bytesBytes)
		} else {
			p.buf.WriteByte('[')
		}
		for i, c := range v {
			if i > 0 {
				if p.fmt.sharp {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			p.printField(c, 'v', p.fmt.plus, p.fmt.sharp, depth+1)
		}
		if sharp {
			p.buf.WriteByte('}')
		} else {
			p.buf.WriteByte(']')
		}
		return
	}
	s := string(v)
	switch verb {
	case 's':
		p.fmt.fmt_s(s)
	case 'x':
		p.fmt.fmt_sx(s)
	case 'X':
		p.fmt.fmt_sX(s)
	case 'q':
		p.fmt.fmt_q(s)
	default:
		p.badVerb(verb, value)
	}
}

func (p *pp) fmtUintptrGetter(field interface{}, value reflect.Value, verb int, sharp bool) bool {
	v, ok := value.(uintptrGetter)
	if !ok {
		return false
	}
	u := v.Get()
	if sharp {
		p.add('(')
		p.buf.WriteString(reflect.Typeof(field).String())
		p.add(')')
		p.add('(')
		if u == 0 {
			p.buf.Write(nilBytes)
		} else {
			p.fmt0x64(uint64(v.Get()))
		}
		p.add(')')
	} else {
		p.fmt0x64(uint64(u))
	}
	return true
}

func (p *pp) printField(field interface{}, verb int, plus, sharp bool, depth int) (was_string bool) {
	if field != nil {
		switch {
		default:
			if stringer, ok := field.(Stringer); ok {
				p.printField(stringer.String(), verb, plus, sharp, depth)
				return false // this value is not a string
			}
		case sharp:
			if stringer, ok := field.(GoStringer); ok {
				p.printField(stringer.GoString(), verb, plus, sharp, depth)
				return false // this value is not a string
			}
		}
	}

	// Some types can be done without reflection.
	switch f := field.(type) {
	case bool:
		p.fmtBool(f, verb, field)
		return false
	case float:
		if floatBits == 32 {
			p.fmtFloat32(float32(f), verb, field)
		} else {
			p.fmtFloat64(float64(f), verb, field)
		}
		return false
	case float32:
		p.fmtFloat32(f, verb, field)
		return false
	case float64:
		p.fmtFloat64(f, verb, field)
		return false
	case complex:
		if complexBits == 64 {
			p.fmtComplex64(complex64(f), verb, field)
		} else {
			p.fmtComplex128(complex128(f), verb, field)
		}
		return false
	case complex64:
		p.fmtComplex64(complex64(f), verb, field)
		return false
	case complex128:
		p.fmtComplex128(f, verb, field)
		return false
	case int:
		p.fmtInt64(int64(f), verb, field)
		return false
	case int8:
		p.fmtInt64(int64(f), verb, field)
		return false
	case int16:
		p.fmtInt64(int64(f), verb, field)
		return false
	case int32:
		p.fmtInt64(int64(f), verb, field)
		return false
	case int64:
		p.fmtInt64(f, verb, field)
		return false
	case uint:
		p.fmtUint64(uint64(f), verb, sharp, field)
		return false
	case uint8:
		p.fmtUint64(uint64(f), verb, sharp, field)
		return false
	case uint16:
		p.fmtUint64(uint64(f), verb, sharp, field)
		return false
	case uint32:
		p.fmtUint64(uint64(f), verb, sharp, field)
		return false
	case uint64:
		p.fmtUint64(f, verb, sharp, field)
		return false
	case uintptr:
		p.fmtUint64(uint64(f), verb, sharp, field)
		return false
	case string:
		p.fmtString(f, verb, sharp, field)
		return verb == 's' || verb == 'v'
	case []byte:
		p.fmtBytes(f, verb, sharp, depth, field)
		return verb == 's'
	}

	if field == nil {
		if verb == 'v' {
			p.buf.Write(nilAngleBytes)
		} else {
			p.badVerb(verb, field)
		}
		return false
	}

	value := reflect.NewValue(field)
	// Need to use reflection
	// Special case for reflection values that know how to print with %p.
	if verb == 'p' && p.fmtUintptrGetter(field, value, verb, sharp) {
		return false
	}

BigSwitch:
	switch f := value.(type) {
	case *reflect.BoolValue:
		p.fmtBool(f.Get(), verb, field)
	case *reflect.IntValue:
		p.fmtInt64(int64(f.Get()), verb, field)
	case *reflect.Int8Value:
		p.fmtInt64(int64(f.Get()), verb, field)
	case *reflect.Int16Value:
		p.fmtInt64(int64(f.Get()), verb, field)
	case *reflect.Int32Value:
		p.fmtInt64(int64(f.Get()), verb, field)
	case *reflect.Int64Value:
		p.fmtInt64(f.Get(), verb, field)
	case *reflect.UintValue:
		p.fmtUint64(uint64(f.Get()), verb, sharp, field)
	case *reflect.Uint8Value:
		p.fmtUint64(uint64(f.Get()), verb, sharp, field)
	case *reflect.Uint16Value:
		p.fmtUint64(uint64(f.Get()), verb, sharp, field)
	case *reflect.Uint32Value:
		p.fmtUint64(uint64(f.Get()), verb, sharp, field)
	case *reflect.Uint64Value:
		p.fmtUint64(f.Get(), verb, sharp, field)
	case *reflect.UintptrValue:
		p.fmtUint64(uint64(f.Get()), verb, sharp, field)
	case *reflect.FloatValue:
		if floatBits == 32 {
			p.fmtFloat32(float32(f.Get()), verb, field)
		} else {
			p.fmtFloat64(float64(f.Get()), verb, field)
		}
	case *reflect.Float32Value:
		p.fmtFloat64(float64(f.Get()), verb, field)
	case *reflect.Float64Value:
		p.fmtFloat64(f.Get(), verb, field)
	case *reflect.ComplexValue:
		if complexBits == 64 {
			p.fmtComplex64(complex64(f.Get()), verb, field)
		} else {
			p.fmtComplex128(complex128(f.Get()), verb, field)
		}
	case *reflect.Complex64Value:
		p.fmtComplex64(f.Get(), verb, field)
	case *reflect.Complex128Value:
		p.fmtComplex128(f.Get(), verb, field)
	case *reflect.StringValue:
		p.fmtString(f.Get(), verb, sharp, field)
	case *reflect.MapValue:
		if sharp {
			p.buf.WriteString(f.Type().String())
			p.buf.WriteByte('{')
		} else {
			p.buf.Write(mapBytes)
		}
		keys := f.Keys()
		for i, key := range keys {
			if i > 0 {
				if sharp {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			p.printField(key.Interface(), verb, plus, sharp, depth+1)
			p.buf.WriteByte(':')
			p.printField(f.Elem(key).Interface(), verb, plus, sharp, depth+1)
		}
		if sharp {
			p.buf.WriteByte('}')
		} else {
			p.buf.WriteByte(']')
		}
	case *reflect.StructValue:
		if sharp {
			p.buf.WriteString(reflect.Typeof(field).String())
		}
		p.add('{')
		v := f
		t := v.Type().(*reflect.StructType)
		p.fmt.clearflags() // clear flags for p.printField
		for i := 0; i < v.NumField(); i++ {
			if i > 0 {
				if sharp {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			if plus || sharp {
				if f := t.Field(i); f.Name != "" {
					p.buf.WriteString(f.Name)
					p.buf.WriteByte(':')
				}
			}
			p.printField(getField(v, i).Interface(), verb, plus, sharp, depth+1)
		}
		p.buf.WriteByte('}')
	case *reflect.InterfaceValue:
		value := f.Elem()
		if value == nil {
			if sharp {
				p.buf.WriteString(reflect.Typeof(field).String())
				p.buf.Write(nilParenBytes)
			} else {
				p.buf.Write(nilAngleBytes)
			}
		} else {
			return p.printField(value.Interface(), verb, plus, sharp, depth+1)
		}
	case reflect.ArrayOrSliceValue:
		if sharp {
			p.buf.WriteString(reflect.Typeof(field).String())
			p.buf.WriteByte('{')
		} else {
			p.buf.WriteByte('[')
		}
		for i := 0; i < f.Len(); i++ {
			if i > 0 {
				if sharp {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			p.printField(f.Elem(i).Interface(), verb, plus, sharp, depth+1)
		}
		if sharp {
			p.buf.WriteByte('}')
		} else {
			p.buf.WriteByte(']')
		}
	case *reflect.PtrValue:
		v := f.Get()
		// pointer to array or slice or struct?  ok at top level
		// but not embedded (avoid loops)
		if v != 0 && depth == 0 {
			switch a := f.Elem().(type) {
			case reflect.ArrayOrSliceValue:
				p.buf.WriteByte('&')
				p.printField(a.Interface(), verb, plus, sharp, depth+1)
				break BigSwitch
			case *reflect.StructValue:
				p.buf.WriteByte('&')
				p.printField(a.Interface(), verb, plus, sharp, depth+1)
				break BigSwitch
			}
		}
		if sharp {
			p.buf.WriteByte('(')
			p.buf.WriteString(reflect.Typeof(field).String())
			p.buf.WriteByte(')')
			p.buf.WriteByte('(')
			if v == 0 {
				p.buf.Write(nilBytes)
			} else {
				p.fmt0x64(uint64(v))
			}
			p.buf.WriteByte(')')
			break
		}
		if v == 0 {
			p.buf.Write(nilAngleBytes)
			break
		}
		p.fmt0x64(uint64(v))
	case uintptrGetter:
		if p.fmtUintptrGetter(field, value, verb, sharp) {
			break
		}
		p.unknownType(f)
	default:
		p.unknownType(f)
	}
	return false
}

func (p *pp) doPrintf(format string, a []interface{}) {
	end := len(format) - 1
	fieldnum := 0 // we process one field per non-trivial format
	for i := 0; i <= end; {
		c, w := utf8.DecodeRuneInString(format[i:])
		if c != '%' || i == end {
			if w == 1 {
				p.buf.WriteByte(byte(c))
			} else {
				p.buf.WriteString(format[i : i+w])
			}
			i += w
			continue
		}
		i++
		// flags and widths
		p.fmt.clearflags()
	F:
		for ; i < end; i++ {
			switch format[i] {
			case '#':
				p.fmt.sharp = true
			case '0':
				p.fmt.zero = true
			case '+':
				p.fmt.plus = true
			case '-':
				p.fmt.minus = true
			case ' ':
				p.fmt.space = true
			default:
				break F
			}
		}
		// do we have 20 (width)?
		p.fmt.wid, p.fmt.widPresent, i = parsenum(format, i, end)
		// do we have .20 (precision)?
		if i < end && format[i] == '.' {
			p.fmt.prec, p.fmt.precPresent, i = parsenum(format, i+1, end)
		}
		c, w = utf8.DecodeRuneInString(format[i:])
		i += w
		// percent is special - absorbs no operand
		if c == '%' {
			p.buf.WriteByte('%') // TODO: should we bother with width & prec?
			continue
		}
		if fieldnum >= len(a) { // out of operands
			p.buf.WriteByte('%')
			p.add(c)
			p.buf.Write(missingBytes)
			continue
		}
		field := a[fieldnum]
		fieldnum++

		// %T is special; we always do it here.
		if c == 'T' {
			// the value's type
			if field == nil {
				p.buf.Write(nilAngleBytes)
				break
			}
			p.buf.WriteString(reflect.Typeof(field).String())
			continue
		}

		// Try Formatter (except for %T).
		if field != nil {
			if formatter, ok := field.(Formatter); ok {
				formatter.Format(p, c)
				continue
			}
		}

		p.printField(field, c, p.fmt.plus, p.fmt.sharp, 0)
	}

	if fieldnum < len(a) {
		p.buf.Write(extraBytes)
		for ; fieldnum < len(a); fieldnum++ {
			field := a[fieldnum]
			if field != nil {
				p.buf.WriteString(reflect.Typeof(field).String())
				p.buf.WriteByte('=')
			}
			p.printField(field, 'v', false, false, 0)
			if fieldnum+1 < len(a) {
				p.buf.Write(commaSpaceBytes)
			}
		}
		p.buf.WriteByte(')')
	}
}

func (p *pp) doPrint(a []interface{}, addspace, addnewline bool) {
	prev_string := false
	for fieldnum := 0; fieldnum < len(a); fieldnum++ {
		// always add spaces if we're doing println
		field := a[fieldnum]
		if fieldnum > 0 {
			_, is_string := field.(*reflect.StringValue)
			if addspace || !is_string && !prev_string {
				p.buf.WriteByte(' ')
			}
		}
		prev_string = p.printField(field, 'v', false, false, 0)
	}
	if addnewline {
		p.buf.WriteByte('\n')
	}
}
