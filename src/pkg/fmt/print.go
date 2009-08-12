// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	Package fmt implements formatted I/O with functions analogous
	to C's printf.  The format 'verbs' are derived from C's but
	are simpler.

	The verbs:

	General:
		%v	for any operand type, the value in a default format.
			when printing structs, the plus flag (%+v) adds field names
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
		%f	decimal point but no exponent, e.g. 123.456
		%g	whichever of %e or %f produces more compact output
	String and slice of bytes:
		%s	the uninterpreted bytes of the string or slice
		%q	a double-quoted string safely escaped with Go syntax
		%x	base 16 notation with two characters per byte
	Pointer:
		%p	base 16 notation, with leading 0x
	Type:
		%T	a Go-syntax representation of the type of the operand

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
	"io";
	"os";
	"reflect";
	"utf8";
)

// State represents the printer state passed to custom formatters.
// It provides access to the io.Writer interface plus information about
// the flags and options for the operand's format specifier.
type State interface {
	// Write is the function to call to emit formatted output to be printed.
	Write(b []byte) (ret int, err os.Error);
	// Width returns the value of the width option and whether it has been set.
	Width()	(wid int, ok bool);
	// Precision returns the value of the precision option and whether it has been set.
	Precision()	(prec int, ok bool);

	// Flag returns whether the flag c, a character, has been set.
	Flag(int)	bool;
}

// Formatter is the interface implemented by objects with a custom formatter.
// The implementation of Format may call Sprintf or Fprintf(f) etc.
// to generate its output.
type Formatter interface {
	Format(f State, c int);
}

// String represents any object being printed that has a String() method that
// returns a string, which defines the ``native'' format for that object.
// Any such object will be printed using that method if passed
// as operand to a %s or %v format or to an unformatted printer such as Print.
type Stringer interface {
	String() string
}

const runeSelf = utf8.RuneSelf
const allocSize = 32

type pp struct {
	n	int;
	buf	[]byte;
	fmt	*Fmt;
}

func newPrinter() *pp {
	p := new(pp);
	p.fmt = New();
	return p;
}

func (p *pp) Width() (wid int, ok bool) {
	return p.fmt.wid, p.fmt.wid_present
}

func (p *pp) Precision() (prec int, ok bool) {
	return p.fmt.prec, p.fmt.prec_present
}

func (p *pp) Flag(b int) bool {
	switch b {
	case '-':
		return p.fmt.minus;
	case '+':
		return p.fmt.plus;
	case '#':
		return p.fmt.sharp;
	case ' ':
		return p.fmt.space;
	case '0':
		return p.fmt.zero;
	}
	return false
}

func (p *pp) ensure(n int) {
	if len(p.buf) < n {
		newn := allocSize + len(p.buf);
		if newn < n {
			newn = n + allocSize
		}
		b := make([]byte, newn);
		for i := 0; i < p.n; i++ {
			b[i] = p.buf[i];
		}
		p.buf = b;
	}
}

func (p *pp) addstr(s string) {
	n := len(s);
	p.ensure(p.n + n);
	for i := 0; i < n; i++ {
		p.buf[p.n] = s[i];
		p.n++;
	}
}

func (p *pp) addbytes(b []byte, start, end int) {
	p.ensure(p.n + end-start);
	for i := start; i < end; i++ {
		p.buf[p.n] = b[i];
		p.n++;
	}
}

func (p *pp) add(c int) {
	p.ensure(p.n + 1);
	if c < runeSelf {
		p.buf[p.n] = byte(c);
		p.n++;
	} else {
		p.addstr(string(c));
	}
}

// Implement Write so we can call fprintf on a P, for
// recursive use in custom verbs.
func (p *pp) Write(b []byte) (ret int, err os.Error) {
	p.addbytes(b, 0, len(b));
	return len(b), nil;
}

// These routines end in 'f' and take a format string.

// Fprintf formats according to a format specifier and writes to w.
func Fprintf(w io.Writer, format string, a ...) (n int, error os.Error) {
	v := reflect.NewValue(a).(*reflect.StructValue);
	p := newPrinter();
	p.doprintf(format, v);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

// Printf formats according to a format specifier and writes to standard output.
func Printf(format string, v ...) (n int, errno os.Error) {
	n, errno = Fprintf(os.Stdout, format, v);
	return n, errno;
}

// Sprintf formats according to a format specifier and returns the resulting string.
func Sprintf(format string, a ...) string {
	v := reflect.NewValue(a).(*reflect.StructValue);
	p := newPrinter();
	p.doprintf(format, v);
	s := string(p.buf)[0 : p.n];
	return s;
}

// These routines do not take a format string

// Fprint formats using the default formats for its operands and writes to w.
// Spaces are added between operands when neither is a string.
func Fprint(w io.Writer, a ...) (n int, error os.Error) {
	v := reflect.NewValue(a).(*reflect.StructValue);
	p := newPrinter();
	p.doprint(v, false, false);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

// Print formats using the default formats for its operands and writes to standard output.
// Spaces are added between operands when neither is a string.
func Print(v ...) (n int, errno os.Error) {
	n, errno = Fprint(os.Stdout, v);
	return n, errno;
}

// Sprint formats using the default formats for its operands and returns the resulting string.
// Spaces are added between operands when neither is a string.
func Sprint(a ...) string {
	v := reflect.NewValue(a).(*reflect.StructValue);
	p := newPrinter();
	p.doprint(v, false, false);
	s := string(p.buf)[0 : p.n];
	return s;
}

// These routines end in 'ln', do not take a format string,
// always add spaces between operands, and add a newline
// after the last operand.

// Fprintln formats using the default formats for its operands and writes to w.
// Spaces are always added between operands and a newline is appended.
func Fprintln(w io.Writer, a ...) (n int, error os.Error) {
	v := reflect.NewValue(a).(*reflect.StructValue);
	p := newPrinter();
	p.doprint(v, true, true);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

// Println formats using the default formats for its operands and writes to standard output.
// Spaces are always added between operands and a newline is appended.
func Println(v ...) (n int, errno os.Error) {
	n, errno = Fprintln(os.Stdout, v);
	return n, errno;
}

// Sprintln formats using the default formats for its operands and returns the resulting string.
// Spaces are always added between operands and a newline is appended.
func Sprintln(a ...) string {
	v := reflect.NewValue(a).(*reflect.StructValue);
	p := newPrinter();
	p.doprint(v, true, true);
	s := string(p.buf)[0 : p.n];
	return s;
}


// Get the i'th arg of the struct value.
// If the arg itself is an interface, return a value for
// the thing inside the interface, not the interface itself.
func getField(v *reflect.StructValue, i int) reflect.Value {
	val := v.Field(i);
	if i, ok := val.(*reflect.InterfaceValue); ok {
		if inter := i.Interface(); inter != nil {
			return reflect.NewValue(inter);
		}
	}
	return val;
}

// Getters for the fields of the argument structure.

func getBool(v reflect.Value) (val bool, ok bool) {
	if b, ok := v.(*reflect.BoolValue); ok {
		return b.Get(), true;
	}
	return;
}

func getInt(v reflect.Value) (val int64, signed, ok bool) {
	switch v := v.(type) {
	case *reflect.IntValue:
		return int64(v.Get()), true, true;
	case *reflect.Int8Value:
		return int64(v.Get()), true, true;
	case *reflect.Int16Value:
		return int64(v.Get()), true, true;
	case *reflect.Int32Value:
		return int64(v.Get()), true, true;
	case *reflect.Int64Value:
		return int64(v.Get()), true, true;
	case *reflect.UintValue:
		return int64(v.Get()), false, true;
	case *reflect.Uint8Value:
		return int64(v.Get()), false, true;
	case *reflect.Uint16Value:
		return int64(v.Get()), false, true;
	case *reflect.Uint32Value:
		return int64(v.Get()), false, true;
	case *reflect.Uint64Value:
		return int64(v.Get()), false, true;
	case *reflect.UintptrValue:
		return int64(v.Get()), false, true;
	}
	return;
}

func getString(v reflect.Value) (val string, ok bool) {
	if v, ok := v.(*reflect.StringValue); ok {
		return v.Get(), true;
	}
	if bytes, ok := v.Interface().([]byte); ok {
		return string(bytes), true;
	}
	return;
}

func getFloat32(v reflect.Value) (val float32, ok bool) {
	switch v := v.(type) {
	case *reflect.Float32Value:
		return float32(v.Get()), true;
	case *reflect.FloatValue:
		if v.Type().Size()*8 == 32 {
			return float32(v.Get()), true;
		}
	}
	return;
}

func getFloat64(v reflect.Value) (val float64, ok bool) {
	switch v := v.(type) {
	case *reflect.FloatValue:
		if v.Type().Size()*8 == 64 {
			return float64(v.Get()), true;
		}
	case *reflect.Float64Value:
		return float64(v.Get()), true;
	}
	return;
}

func getPtr(v reflect.Value) (val uintptr, ok bool) {
	switch v := v.(type) {
	case *reflect.PtrValue:
		return uintptr(v.Get()), true;
	}
	return;
}

// Convert ASCII to integer.  n is 0 (and got is false) if no number present.

func parsenum(s string, start, end int) (n int, got bool, newi int) {
	if start >= end {
		return 0, false, end
	}
	isnum := false;
	num := 0;
	for '0' <= s[start] && s[start] <= '9' {
		num = num*10 + int(s[start] - '0');
		start++;
		isnum = true;
	}
	return num, isnum, start;
}

func (p *pp) printField(field reflect.Value) (was_string bool) {
	inter := field.Interface();
	if inter != nil {
		if stringer, ok := inter.(Stringer); ok {
			p.addstr(stringer.String());
			return false;	// this value is not a string
		}
	}
	s := "";
	switch f := field.(type) {
	case *reflect.BoolValue:
		s = p.fmt.Fmt_boolean(f.Get()).Str();
	case *reflect.Float32Value:
		s = p.fmt.Fmt_g32(f.Get()).Str();
	case *reflect.Float64Value:
		s = p.fmt.Fmt_g64(f.Get()).Str();
	case *reflect.FloatValue:
		if field.Type().Size()*8 == 32 {
			s = p.fmt.Fmt_g32(float32(f.Get())).Str();
		} else {
			s = p.fmt.Fmt_g64(float64(f.Get())).Str();
		}
	case *reflect.StringValue:
		s = p.fmt.Fmt_s(f.Get()).Str();
		was_string = true;
	case *reflect.PtrValue:
		v := f.Get();
		if v == 0 {
			s = "<nil>";
			break;
		}
		// pointer to array?
		if a, ok := f.Elem().(reflect.ArrayOrSliceValue); ok {
			p.addstr("&");
			p.printField(a);
			break;
		}
		p.fmt.sharp = !p.fmt.sharp;  // turn 0x on by default
		s = p.fmt.Fmt_ux64(uint64(v)).Str();
	case reflect.ArrayOrSliceValue:
		p.addstr("[");
		for i := 0; i < f.Len(); i++ {
			if i > 0 {
				p.addstr(" ");
			}
			p.printField(f.Elem(i));
		}
		p.addstr("]");
	case *reflect.MapValue:
		p.addstr("map[");
		keys := f.Keys();
		for i, key := range keys {
			if i > 0 {
				p.addstr(" ");
			}
			p.printField(key);
			p.addstr(":");
			p.printField(f.Elem(key));
		}
		p.addstr("]");
	case *reflect.StructValue:
		p.add('{');
		v := f;
		t := v.Type().(*reflect.StructType);
		donames := p.fmt.plus;
		p.fmt.clearflags();	// clear flags for p.printField
		for i := 0; i < v.NumField();  i++ {
			if i > 0 {
				p.add(' ')
			}
			if donames {
				if f := t.Field(i); f.Name != "" {
					p.addstr(f.Name);
					p.add(':');
				}
			}
			p.printField(getField(v, i));
		}
		p.add('}');
	case *reflect.InterfaceValue:
		value := f.Elem();
		if value == nil {
			s = "<nil>"
		} else {
			return p.printField(value);
		}
	case *reflect.UintptrValue:
		p.fmt.sharp = !p.fmt.sharp;  // turn 0x on by default
		s = p.fmt.Fmt_ux64(uint64(f.Get())).Str();
	default:
		v, signed, ok := getInt(field);
		if ok {
			if signed {
				s = p.fmt.Fmt_d64(v).Str();
			} else {
				s = p.fmt.Fmt_ud64(uint64(v)).Str();
			}
			break;
		}
		s = "?" + field.Type().String() + "?";
	}
	p.addstr(s);
	return was_string;
}

func (p *pp) doprintf(format string, v *reflect.StructValue) {
	p.ensure(len(format));	// a good starting size
	end := len(format) - 1;
	fieldnum := 0;	// we process one field per non-trivial format
	for i := 0; i <= end;  {
		c, w := utf8.DecodeRuneInString(format[i:len(format)]);
		if c != '%' || i == end {
			p.add(c);
			i += w;
			continue;
		}
		i++;
		// flags and widths
		p.fmt.clearflags();
		F: for ; i < end; i++ {
			switch format[i] {
			case '#':
				p.fmt.sharp = true;
			case '0':
				p.fmt.zero = true;
			case '+':
				p.fmt.plus = true;
			case '-':
				p.fmt.minus = true;
			case ' ':
				p.fmt.space = true;
			default:
				break F;
			}
		}
		// do we have 20 (width)?
		p.fmt.wid, p.fmt.wid_present, i = parsenum(format, i, end);
		// do we have .20 (precision)?
		if i < end && format[i] == '.' {
			p.fmt.prec, p.fmt.prec_present, i = parsenum(format, i+1, end);
		}
		c, w = utf8.DecodeRuneInString(format[i:len(format)]);
		i += w;
		// percent is special - absorbs no operand
		if c == '%' {
			p.add('%');	// TODO: should we bother with width & prec?
			continue;
		}
		if fieldnum >= v.NumField() {	// out of operands
			p.add('%');
			p.add(c);
			p.addstr("(missing)");
			continue;
		}
		field := getField(v, fieldnum);
		fieldnum++;
		inter := field.Interface();
		if inter != nil && c != 'T' {	// don't want thing to describe itself if we're asking for its type
			if formatter, ok := inter.(Formatter); ok {
				formatter.Format(p, c);
				continue;
			}
		}
		s := "";
		switch c {
			// bool
			case 't':
				if v, ok := getBool(field); ok {
					if v {
						s = "true";
					} else {
						s = "false";
					}
				} else {
					goto badtype;
				}

			// int
			case 'b':
				if v, signed, ok := getInt(field); ok {
					s = p.fmt.Fmt_b64(uint64(v)).Str()	// always unsigned
				} else if v, ok := getFloat32(field); ok {
					s = p.fmt.Fmt_fb32(v).Str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.Fmt_fb64(v).Str()
				} else {
					goto badtype
				}
			case 'c':
				if v, signed, ok := getInt(field); ok {
					s = p.fmt.Fmt_c(int(v)).Str()
				} else {
					goto badtype
				}
			case 'd':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.Fmt_d64(v).Str()
					} else {
						s = p.fmt.Fmt_ud64(uint64(v)).Str()
					}
				} else {
					goto badtype
				}
			case 'o':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.Fmt_o64(v).Str()
					} else {
						s = p.fmt.Fmt_uo64(uint64(v)).Str()
					}
				} else {
					goto badtype
				}
			case 'x':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.Fmt_x64(v).Str()
					} else {
						s = p.fmt.Fmt_ux64(uint64(v)).Str()
					}
				} else if v, ok := getString(field); ok {
					s = p.fmt.Fmt_sx(v).Str();
				} else {
					goto badtype
				}
			case 'X':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.Fmt_X64(v).Str()
					} else {
						s = p.fmt.Fmt_uX64(uint64(v)).Str()
					}
				} else if v, ok := getString(field); ok {
					s = p.fmt.Fmt_sX(v).Str();
				} else {
					goto badtype
				}

			// float
			case 'e':
				if v, ok := getFloat32(field); ok {
					s = p.fmt.Fmt_e32(v).Str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.Fmt_e64(v).Str()
				} else {
					goto badtype
				}
			case 'f':
				if v, ok := getFloat32(field); ok {
					s = p.fmt.Fmt_f32(v).Str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.Fmt_f64(v).Str()
				} else {
					goto badtype
				}
			case 'g':
				if v, ok := getFloat32(field); ok {
					s = p.fmt.Fmt_g32(v).Str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.Fmt_g64(v).Str()
				} else {
					goto badtype
				}

			// string
			case 's':
				if inter != nil {
					// if object implements String, use the result.
					if stringer, ok := inter.(Stringer); ok {
						s = p.fmt.Fmt_s(stringer.String()).Str();
						break;
					}
				}
				if v, ok := getString(field); ok {
					s = p.fmt.Fmt_s(v).Str()
				} else {
					goto badtype
				}
			case 'q':
				if v, ok := getString(field); ok {
					s = p.fmt.Fmt_q(v).Str()
				} else {
					goto badtype
				}

			// pointer
			case 'p':
				if v, ok := getPtr(field); ok {
					if v == 0 {
						s = "<nil>"
					} else {
						s = "0x" + p.fmt.Fmt_uX64(uint64(v)).Str()
					}
				} else {
					goto badtype
				}

			// arbitrary value; do your best
			case 'v':
				p.printField(field);

			// the value's type
			case 'T':
				s = field.Type().String();

			default:
			badtype:
				s = "%" + string(c) + "(" + field.Type().String() + ")%";
		}
		p.addstr(s);
	}
	if fieldnum < v.NumField() {
		p.addstr("?(extra ");
		for ; fieldnum < v.NumField(); fieldnum++ {
			p.addstr(getField(v, fieldnum).Type().String());
			if fieldnum + 1 < v.NumField() {
				p.addstr(", ");
			}
		}
		p.addstr(")");
	}
}

func (p *pp) doprint(v *reflect.StructValue, addspace, addnewline bool) {
	prev_string := false;
	for fieldnum := 0; fieldnum < v.NumField();  fieldnum++ {
		// always add spaces if we're doing println
		field := getField(v, fieldnum);
		if fieldnum > 0 {
			_, is_string := field.(*reflect.StringValue);
			if addspace || !is_string && !prev_string {
				p.add(' ');
			}
		}
		prev_string = p.printField(field);
	}
	if addnewline {
		p.add('\n')
	}
}
