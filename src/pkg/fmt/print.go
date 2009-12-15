// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	Package fmt implements formatted I/O with functions analogous
	to C's printf.  The format 'verbs' are derived from C's but
	are simpler.

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
	Floating-point:
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
	format %6.2f prints 123.45.

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
	will be used for %v, %s, or Print etc.
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
	trueBytes       = []byte{'t', 'r', 'u', 'e'}
	falseBytes      = []byte{'f', 'a', 'l', 's', 'e'}
	commaSpaceBytes = []byte{',', ' '}
	nilAngleBytes   = []byte{'<', 'n', 'i', 'l', '>'}
	nilParenBytes   = []byte{'(', 'n', 'i', 'l', ')'}
	nilBytes        = []byte{'n', 'i', 'l'}
	mapBytes        = []byte{'m', 'a', 'p', '['}
	missingBytes    = []byte{'m', 'i', 's', 's', 'i', 'n', 'g'}
	extraBytes      = []byte{'?', '(', 'e', 'x', 't', 'r', 'a', ' '}
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

const allocSize = 32

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
		w := utf8.EncodeRune(c, &p.runeBuf)
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
func Fprintf(w io.Writer, format string, a ...) (n int, error os.Error) {
	v := reflect.NewValue(a).(*reflect.StructValue)
	p := newPrinter()
	p.doprintf(format, v)
	n64, error := p.buf.WriteTo(w)
	p.free()
	return int(n64), error
}

// Printf formats according to a format specifier and writes to standard output.
func Printf(format string, v ...) (n int, errno os.Error) {
	n, errno = Fprintf(os.Stdout, format, v)
	return n, errno
}

// Sprintf formats according to a format specifier and returns the resulting string.
func Sprintf(format string, a ...) string {
	v := reflect.NewValue(a).(*reflect.StructValue)
	p := newPrinter()
	p.doprintf(format, v)
	s := p.buf.String()
	p.free()
	return s
}

// These routines do not take a format string

// Fprint formats using the default formats for its operands and writes to w.
// Spaces are added between operands when neither is a string.
func Fprint(w io.Writer, a ...) (n int, error os.Error) {
	v := reflect.NewValue(a).(*reflect.StructValue)
	p := newPrinter()
	p.doprint(v, false, false)
	n64, error := p.buf.WriteTo(w)
	p.free()
	return int(n64), error
}

// Print formats using the default formats for its operands and writes to standard output.
// Spaces are added between operands when neither is a string.
func Print(v ...) (n int, errno os.Error) {
	n, errno = Fprint(os.Stdout, v)
	return n, errno
}

// Sprint formats using the default formats for its operands and returns the resulting string.
// Spaces are added between operands when neither is a string.
func Sprint(a ...) string {
	v := reflect.NewValue(a).(*reflect.StructValue)
	p := newPrinter()
	p.doprint(v, false, false)
	s := p.buf.String()
	p.free()
	return s
}

// These routines end in 'ln', do not take a format string,
// always add spaces between operands, and add a newline
// after the last operand.

// Fprintln formats using the default formats for its operands and writes to w.
// Spaces are always added between operands and a newline is appended.
func Fprintln(w io.Writer, a ...) (n int, error os.Error) {
	v := reflect.NewValue(a).(*reflect.StructValue)
	p := newPrinter()
	p.doprint(v, true, true)
	n64, error := p.buf.WriteTo(w)
	p.free()
	return int(n64), error
}

// Println formats using the default formats for its operands and writes to standard output.
// Spaces are always added between operands and a newline is appended.
func Println(v ...) (n int, errno os.Error) {
	n, errno = Fprintln(os.Stdout, v)
	return n, errno
}

// Sprintln formats using the default formats for its operands and returns the resulting string.
// Spaces are always added between operands and a newline is appended.
func Sprintln(a ...) string {
	v := reflect.NewValue(a).(*reflect.StructValue)
	p := newPrinter()
	p.doprint(v, true, true)
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

// Getters for the fields of the argument structure.

func getBool(v reflect.Value) (val bool, ok bool) {
	if b, ok := v.(*reflect.BoolValue); ok {
		return b.Get(), true
	}
	return
}

func getInt(v reflect.Value) (val int64, signed, ok bool) {
	switch v := v.(type) {
	case *reflect.IntValue:
		return int64(v.Get()), true, true
	case *reflect.Int8Value:
		return int64(v.Get()), true, true
	case *reflect.Int16Value:
		return int64(v.Get()), true, true
	case *reflect.Int32Value:
		return int64(v.Get()), true, true
	case *reflect.Int64Value:
		return int64(v.Get()), true, true
	case *reflect.UintValue:
		return int64(v.Get()), false, true
	case *reflect.Uint8Value:
		return int64(v.Get()), false, true
	case *reflect.Uint16Value:
		return int64(v.Get()), false, true
	case *reflect.Uint32Value:
		return int64(v.Get()), false, true
	case *reflect.Uint64Value:
		return int64(v.Get()), false, true
	case *reflect.UintptrValue:
		return int64(v.Get()), false, true
	}
	return
}

func getString(v reflect.Value) (val string, ok bool) {
	if v, ok := v.(*reflect.StringValue); ok {
		return v.Get(), true
	}
	if bytes, ok := v.Interface().([]byte); ok {
		return string(bytes), true
	}
	return
}

func getFloat32(v reflect.Value) (val float32, ok bool) {
	switch v := v.(type) {
	case *reflect.Float32Value:
		return float32(v.Get()), true
	case *reflect.FloatValue:
		if v.Type().Size()*8 == 32 {
			return float32(v.Get()), true
		}
	}
	return
}

func getFloat64(v reflect.Value) (val float64, ok bool) {
	switch v := v.(type) {
	case *reflect.FloatValue:
		if v.Type().Size()*8 == 64 {
			return float64(v.Get()), true
		}
	case *reflect.Float64Value:
		return float64(v.Get()), true
	}
	return
}

func getPtr(v reflect.Value) (val uintptr, ok bool) {
	switch v := v.(type) {
	case *reflect.PtrValue:
		return uintptr(v.Get()), true
	}
	return
}

// Convert ASCII to integer.  n is 0 (and got is false) if no number present.

func parsenum(s string, start, end int) (n int, got bool, newi int) {
	if start >= end {
		return 0, false, end
	}
	isnum := false
	num := 0
	for '0' <= s[start] && s[start] <= '9' {
		num = num*10 + int(s[start]-'0')
		start++
		isnum = true
	}
	return num, isnum, start
}

type uintptrGetter interface {
	Get() uintptr
}

func (p *pp) printField(field reflect.Value, plus, sharp bool, depth int) (was_string bool) {
	inter := field.Interface()
	if inter != nil {
		switch {
		default:
			if stringer, ok := inter.(Stringer); ok {
				p.buf.WriteString(stringer.String())
				return false // this value is not a string
			}
		case sharp:
			if stringer, ok := inter.(GoStringer); ok {
				p.buf.WriteString(stringer.GoString())
				return false // this value is not a string
			}
		}
	}
BigSwitch:
	switch f := field.(type) {
	case *reflect.BoolValue:
		p.fmt.fmt_boolean(f.Get())
	case *reflect.Float32Value:
		p.fmt.fmt_g32(f.Get())
	case *reflect.Float64Value:
		p.fmt.fmt_g64(f.Get())
	case *reflect.FloatValue:
		if field.Type().Size()*8 == 32 {
			p.fmt.fmt_g32(float32(f.Get()))
		} else {
			p.fmt.fmt_g64(float64(f.Get()))
		}
	case *reflect.StringValue:
		if sharp {
			p.fmt.fmt_q(f.Get())
		} else {
			p.fmt.fmt_s(f.Get())
			was_string = true
		}
	case *reflect.MapValue:
		if sharp {
			p.buf.WriteString(field.Type().String())
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
			p.printField(key, plus, sharp, depth+1)
			p.buf.WriteByte(':')
			p.printField(f.Elem(key), plus, sharp, depth+1)
		}
		if sharp {
			p.buf.WriteByte('}')
		} else {
			p.buf.WriteByte(']')
		}
	case *reflect.StructValue:
		if sharp {
			p.buf.WriteString(field.Type().String())
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
			p.printField(getField(v, i), plus, sharp, depth+1)
		}
		p.buf.WriteByte('}')
	case *reflect.InterfaceValue:
		value := f.Elem()
		if value == nil {
			if sharp {
				p.buf.WriteString(field.Type().String())
				p.buf.Write(nilParenBytes)
			} else {
				p.buf.Write(nilAngleBytes)
			}
		} else {
			return p.printField(value, plus, sharp, depth+1)
		}
	case reflect.ArrayOrSliceValue:
		if sharp {
			p.buf.WriteString(field.Type().String())
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
			p.printField(f.Elem(i), plus, sharp, depth+1)
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
				p.printField(a, plus, sharp, depth+1)
				break BigSwitch
			case *reflect.StructValue:
				p.buf.WriteByte('&')
				p.printField(a, plus, sharp, depth+1)
				break BigSwitch
			}
		}
		if sharp {
			p.buf.WriteByte('(')
			p.buf.WriteString(field.Type().String())
			p.buf.WriteByte(')')
			p.buf.WriteByte('(')
			if v == 0 {
				p.buf.Write(nilBytes)
			} else {
				p.fmt.sharp = true
				p.fmt.fmt_ux64(uint64(v))
			}
			p.buf.WriteByte(')')
			break
		}
		if v == 0 {
			p.buf.Write(nilAngleBytes)
			break
		}
		p.fmt.sharp = true // turn 0x on
		p.fmt.fmt_ux64(uint64(v))
	case uintptrGetter:
		v := f.Get()
		if sharp {
			p.buf.WriteByte('(')
			p.buf.WriteString(field.Type().String())
			p.buf.WriteByte(')')
			p.buf.WriteByte('(')
			if v == 0 {
				p.buf.Write(nilBytes)
			} else {
				p.fmt.sharp = true
				p.fmt.fmt_ux64(uint64(v))
			}
			p.buf.WriteByte(')')
		} else {
			p.fmt.sharp = true // turn 0x on
			p.fmt.fmt_ux64(uint64(f.Get()))
		}
	default:
		v, signed, ok := getInt(field)
		if ok {
			if signed {
				p.fmt.fmt_d64(v)
			} else {
				if sharp {
					p.fmt.sharp = true // turn on 0x
					p.fmt.fmt_ux64(uint64(v))
				} else {
					p.fmt.fmt_ud64(uint64(v))
				}
			}
			break
		}
		p.buf.WriteByte('?')
		p.buf.WriteString(field.Type().String())
		p.buf.WriteByte('?')
	}
	return was_string
}

func (p *pp) doprintf(format string, v *reflect.StructValue) {
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
	F: for ; i < end; i++ {
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
		if fieldnum >= v.NumField() { // out of operands
			p.buf.WriteByte('%')
			p.add(c)
			p.buf.Write(missingBytes)
			continue
		}
		field := getField(v, fieldnum)
		fieldnum++

		// Try formatter except for %T,
		// which is special and handled internally.
		inter := field.Interface()
		if inter != nil && c != 'T' {
			if formatter, ok := inter.(Formatter); ok {
				formatter.Format(p, c)
				continue
			}
		}

		switch c {
		// bool
		case 't':
			if v, ok := getBool(field); ok {
				if v {
					p.buf.Write(trueBytes)
				} else {
					p.buf.Write(falseBytes)
				}
			} else {
				goto badtype
			}

		// int
		case 'b':
			if v, _, ok := getInt(field); ok {
				p.fmt.fmt_b64(uint64(v)) // always unsigned
			} else if v, ok := getFloat32(field); ok {
				p.fmt.fmt_fb32(v)
			} else if v, ok := getFloat64(field); ok {
				p.fmt.fmt_fb64(v)
			} else {
				goto badtype
			}
		case 'c':
			if v, _, ok := getInt(field); ok {
				p.fmt.fmt_c(int(v))
			} else {
				goto badtype
			}
		case 'd':
			if v, signed, ok := getInt(field); ok {
				if signed {
					p.fmt.fmt_d64(v)
				} else {
					p.fmt.fmt_ud64(uint64(v))
				}
			} else {
				goto badtype
			}
		case 'o':
			if v, signed, ok := getInt(field); ok {
				if signed {
					p.fmt.fmt_o64(v)
				} else {
					p.fmt.fmt_uo64(uint64(v))
				}
			} else {
				goto badtype
			}
		case 'x':
			if v, signed, ok := getInt(field); ok {
				if signed {
					p.fmt.fmt_x64(v)
				} else {
					p.fmt.fmt_ux64(uint64(v))
				}
			} else if v, ok := getString(field); ok {
				p.fmt.fmt_sx(v)
			} else {
				goto badtype
			}
		case 'X':
			if v, signed, ok := getInt(field); ok {
				if signed {
					p.fmt.fmt_X64(v)
				} else {
					p.fmt.fmt_uX64(uint64(v))
				}
			} else if v, ok := getString(field); ok {
				p.fmt.fmt_sX(v)
			} else {
				goto badtype
			}

		// float
		case 'e':
			if v, ok := getFloat32(field); ok {
				p.fmt.fmt_e32(v)
			} else if v, ok := getFloat64(field); ok {
				p.fmt.fmt_e64(v)
			} else {
				goto badtype
			}
		case 'E':
			if v, ok := getFloat32(field); ok {
				p.fmt.fmt_E32(v)
			} else if v, ok := getFloat64(field); ok {
				p.fmt.fmt_E64(v)
			} else {
				goto badtype
			}
		case 'f':
			if v, ok := getFloat32(field); ok {
				p.fmt.fmt_f32(v)
			} else if v, ok := getFloat64(field); ok {
				p.fmt.fmt_f64(v)
			} else {
				goto badtype
			}
		case 'g':
			if v, ok := getFloat32(field); ok {
				p.fmt.fmt_g32(v)
			} else if v, ok := getFloat64(field); ok {
				p.fmt.fmt_g64(v)
			} else {
				goto badtype
			}
		case 'G':
			if v, ok := getFloat32(field); ok {
				p.fmt.fmt_G32(v)
			} else if v, ok := getFloat64(field); ok {
				p.fmt.fmt_G64(v)
			} else {
				goto badtype
			}

		// string
		case 's':
			if inter != nil {
				// if object implements String, use the result.
				if stringer, ok := inter.(Stringer); ok {
					p.fmt.fmt_s(stringer.String())
					break
				}
			}
			if v, ok := getString(field); ok {
				p.fmt.fmt_s(v)
			} else {
				goto badtype
			}
		case 'q':
			if v, ok := getString(field); ok {
				p.fmt.fmt_q(v)
			} else {
				goto badtype
			}

		// pointer
		case 'p':
			if v, ok := getPtr(field); ok {
				if v == 0 {
					p.buf.Write(nilAngleBytes)
				} else {
					p.fmt.fmt_s("0x")
					p.fmt.fmt_uX64(uint64(v))
				}
			} else {
				goto badtype
			}

		// arbitrary value; do your best
		case 'v':
			plus, sharp := p.fmt.plus, p.fmt.sharp
			p.fmt.plus = false
			p.fmt.sharp = false
			p.printField(field, plus, sharp, 0)

		// the value's type
		case 'T':
			p.buf.WriteString(field.Type().String())

		default:
		badtype:
			p.buf.WriteByte('%')
			p.add(c)
			p.buf.WriteByte('(')
			p.buf.WriteString(field.Type().String())
			p.buf.WriteByte('=')
			p.printField(field, false, false, 0)
			p.buf.WriteByte(')')
		}
	}
	if fieldnum < v.NumField() {
		p.buf.Write(extraBytes)
		for ; fieldnum < v.NumField(); fieldnum++ {
			field := getField(v, fieldnum)
			p.buf.WriteString(field.Type().String())
			p.buf.WriteByte('=')
			p.printField(field, false, false, 0)
			if fieldnum+1 < v.NumField() {
				p.buf.Write(commaSpaceBytes)
			}
		}
		p.buf.WriteByte(')')
	}
}

func (p *pp) doprint(v *reflect.StructValue, addspace, addnewline bool) {
	prev_string := false
	for fieldnum := 0; fieldnum < v.NumField(); fieldnum++ {
		// always add spaces if we're doing println
		field := getField(v, fieldnum)
		if fieldnum > 0 {
			_, is_string := field.(*reflect.StringValue)
			if addspace || !is_string && !prev_string {
				p.buf.WriteByte(' ')
			}
		}
		prev_string = p.printField(field, false, false, 0)
	}
	if addnewline {
		p.buf.WriteByte('\n')
	}
}
