// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

/*
	C-like printf, but because of reflection knowledge does not need
	to be told about sizes and signedness (no %llud etc. - just %d).
*/

import (
	"fmt";
	"io";
	"reflect";
	"os";
	"utf8";
)

// Representation of printer state passed to custom formatters.
// Provides access to the io.Write interface plus information about
// the active formatting verb.
type Formatter interface {
	Write(b []byte) (ret int, err *os.Error);
	Width()	(wid int, ok bool);
	Precision()	(prec int, ok bool);

	// flags
	Flag(int)	bool;
}

type Format interface {
	Format(f Formatter, c int);
}

type String interface {
	String() string
}

const runeSelf = 0x80
const allocSize = 32

type pp struct {
	n	int;
	buf	[]byte;
	fmt	*Fmt;
}

func newPrinter() *pp {
	p := new(pp);
	p.fmt = fmt.New();
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
func (p *pp) Write(b []byte) (ret int, err *os.Error) {
	p.addbytes(b, 0, len(b));
	return len(b), nil;
}

func (p *pp) doprintf(format string, v reflect.StructValue);
func (p *pp) doprint(v reflect.StructValue, addspace, addnewline bool);

// These routines end in 'f' and take a format string.

func Fprintf(w io.Write, format string, a ...) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.StructValue);
	p := newPrinter();
	p.doprintf(format, v);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

func Printf(format string, v ...) (n int, errno *os.Error) {
	n, errno = Fprintf(os.Stdout, format, v);
	return n, errno;
}

func Sprintf(format string, a ...) string {
	v := reflect.NewValue(a).(reflect.StructValue);
	p := newPrinter();
	p.doprintf(format, v);
	s := string(p.buf)[0 : p.n];
	return s;
}

// These routines do not take a format string and add spaces only
// when the operand on neither side is a string.

func Fprint(w io.Write, a ...) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.StructValue);
	p := newPrinter();
	p.doprint(v, false, false);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

func Print(v ...) (n int, errno *os.Error) {
	n, errno = Fprint(os.Stdout, v);
	return n, errno;
}

func Sprint(a ...) string {
	v := reflect.NewValue(a).(reflect.StructValue);
	p := newPrinter();
	p.doprint(v, false, false);
	s := string(p.buf)[0 : p.n];
	return s;
}

// These routines end in 'ln', do not take a format string,
// always add spaces between operands, and add a newline
// after the last operand.

func Fprintln(w io.Write, a ...) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.StructValue);
	p := newPrinter();
	p.doprint(v, true, true);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

func Println(v ...) (n int, errno *os.Error) {
	n, errno = Fprintln(os.Stdout, v);
	return n, errno;
}

func Sprintln(a ...) string {
	v := reflect.NewValue(a).(reflect.StructValue);
	p := newPrinter();
	p.doprint(v, true, true);
	s := string(p.buf)[0 : p.n];
	return s;
}


// Get the i'th arg of the struct value.
// If the arg itself is an interface, return a value for
// the thing inside the interface, not the interface itself.
func getField(v reflect.StructValue, i int) reflect.Value {
	val := v.Field(i);
	if val.Kind() == reflect.InterfaceKind {
		inter := val.(reflect.InterfaceValue).Get();
		return reflect.NewValue(inter);
	}
	return val;
}

// Getters for the fields of the argument structure.

func getBool(v reflect.Value) (val bool, ok bool) {
	switch v.Kind() {
	case reflect.BoolKind:
		return v.(reflect.BoolValue).Get(), true;
	}
	return false, false
}

func getInt(v reflect.Value) (val int64, signed, ok bool) {
	switch v.Kind() {
	case reflect.IntKind:
		return int64(v.(reflect.IntValue).Get()), true, true;
	case reflect.Int8Kind:
		return int64(v.(reflect.Int8Value).Get()), true, true;
	case reflect.Int16Kind:
		return int64(v.(reflect.Int16Value).Get()), true, true;
	case reflect.Int32Kind:
		return int64(v.(reflect.Int32Value).Get()), true, true;
	case reflect.Int64Kind:
		return int64(v.(reflect.Int64Value).Get()), true, true;
	case reflect.UintKind:
		return int64(v.(reflect.UintValue).Get()), false, true;
	case reflect.Uint8Kind:
		return int64(v.(reflect.Uint8Value).Get()), false, true;
	case reflect.Uint16Kind:
		return int64(v.(reflect.Uint16Value).Get()), false, true;
	case reflect.Uint32Kind:
		return int64(v.(reflect.Uint32Value).Get()), false, true;
	case reflect.Uint64Kind:
		return int64(v.(reflect.Uint64Value).Get()), false, true;
	case reflect.UintptrKind:
		return int64(v.(reflect.UintptrValue).Get()), false, true;
	}
	return 0, false, false;
}

func getString(v reflect.Value) (val string, ok bool) {
	switch v.Kind() {
	case reflect.StringKind:
		return v.(reflect.StringValue).Get(), true;
	case reflect.ArrayKind:
		if val, ok := v.Interface().([]byte); ok {
			return string(val), true;
		}
	}
	return "", false;
}

func getFloat32(v reflect.Value) (val float32, ok bool) {
	switch v.Kind() {
	case reflect.Float32Kind:
		return float32(v.(reflect.Float32Value).Get()), true;
	case reflect.FloatKind:
		if v.Type().Size()*8 == 32 {
			return float32(v.(reflect.FloatValue).Get()), true;
		}
	}
	return 0.0, false;
}

func getFloat64(v reflect.Value) (val float64, ok bool) {
	switch v.Kind() {
	case reflect.FloatKind:
		if v.Type().Size()*8 == 64 {
			return float64(v.(reflect.FloatValue).Get()), true;
		}
	case reflect.Float64Kind:
		return float64(v.(reflect.Float64Value).Get()), true;
	case reflect.Float80Kind:
		break;	// TODO: what to do here?
	}
	return 0.0, false;
}

func getPtr(v reflect.Value) (val uintptr, ok bool) {
	switch v.Kind() {
	case reflect.PtrKind:
		return uintptr(v.(reflect.PtrValue).Get()), true;
	}
	return 0, false;
}

func getArrayPtr(v reflect.Value) (val reflect.ArrayValue, ok bool) {
	if v.Kind() == reflect.PtrKind {
		v = v.(reflect.PtrValue).Sub();
		if v.Kind() == reflect.ArrayKind {
			return v.(reflect.ArrayValue), true;
		}
	}
	return nil, false;
}

func getArray(v reflect.Value) (val reflect.ArrayValue, ok bool) {
	switch v.Kind() {
	case reflect.ArrayKind:
		return v.(reflect.ArrayValue), true;
	}
	return nil, false;
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
		if stringer, ok := inter.(String); ok {
			p.addstr(stringer.String());
			return false;	// this value is not a string
		}
	}
	s := "";
	switch field.Kind() {
	case reflect.BoolKind:
		s = p.fmt.Fmt_boolean(field.(reflect.BoolValue).Get()).Str();
	case reflect.IntKind, reflect.Int8Kind, reflect.Int16Kind, reflect.Int32Kind, reflect.Int64Kind:
		v, signed, ok := getInt(field);
		s = p.fmt.Fmt_d64(v).Str();
	case reflect.UintKind, reflect.Uint8Kind, reflect.Uint16Kind, reflect.Uint32Kind, reflect.Uint64Kind:
		v, signed, ok := getInt(field);
		s = p.fmt.Fmt_ud64(uint64(v)).Str();
	case reflect.UintptrKind:
		v, signed, ok := getInt(field);
		p.fmt.sharp = !p.fmt.sharp;  // turn 0x on by default
		s = p.fmt.Fmt_ux64(uint64(v)).Str();
	case reflect.Float32Kind:
		v, ok := getFloat32(field);
		s = p.fmt.Fmt_g32(v).Str();
	case reflect.Float64Kind, reflect.Float80Kind:
		v, ok := getFloat64(field);
		s = p.fmt.Fmt_g64(v).Str();
	case reflect.FloatKind:
		if field.Type().Size()*8 == 32 {
			v, ok := getFloat32(field);
			s = p.fmt.Fmt_g32(v).Str();
		} else {
			v, ok := getFloat64(field);
			s = p.fmt.Fmt_g64(v).Str();
		}
	case reflect.StringKind:
		v, ok := getString(field);
		s = p.fmt.Fmt_s(v).Str();
		was_string = true;
	case reflect.PtrKind:
		if v, ok := getPtr(field); v == 0 {
			s = "<nil>"
		} else {
			// pointer to array?  (TODO(r): holdover; delete?)
			if a, ok := getArrayPtr(field); ok {
				p.addstr("&[");
				for i := 0; i < a.Len(); i++ {
					if i > 0 {
						p.addstr(" ");
					}
					p.printField(a.Elem(i));
				}
				p.addstr("]");
			} else {
				p.fmt.sharp = !p.fmt.sharp;  // turn 0x on by default
				s = p.fmt.Fmt_uX64(uint64(v)).Str();
			}
		}
	case reflect.ArrayKind:
		if a, ok := getArray(field); ok {
			p.addstr("[");
			for i := 0; i < a.Len(); i++ {
				if i > 0 {
					p.addstr(" ");
				}
				p.printField(a.Elem(i));
			}
			p.addstr("]");
		}
	case reflect.StructKind:
		p.add('{');
		v := field.(reflect.StructValue);
		t := v.Type().(reflect.StructType);
		donames := p.fmt.plus;
		p.fmt.clearflags();	// clear flags for p.printField
		for i := 0; i < v.Len();  i++ {
			if i > 0 {
				p.add(' ')
			}
			if donames {
				if name, typ, tag, off := t.Field(i); name != "" {
					p.addstr(name);
					p.add('=');
				}
			}
			p.printField(getField(v, i));
		}
		p.add('}');
	case reflect.InterfaceKind:
		inter := field.(reflect.InterfaceValue).Get();
		if inter == nil {
			s = "<nil>"
		} else {
			// should never happen since a non-nil interface always has a type
			s = "<non-nil interface>";
		}
	default:
		s = "?" + field.Type().String() + "?";
	}
	p.addstr(s);
	return was_string;
}

func (p *pp) doprintf(format string, v reflect.StructValue) {
	p.ensure(len(format));	// a good starting size
	end := len(format) - 1;
	fieldnum := 0;	// we process one field per non-trivial format
	for i := 0; i <= end;  {
		c, w := utf8.DecodeRuneInString(format, i);
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
		c, w = utf8.DecodeRuneInString(format, i);
		i += w;
		// percent is special - absorbs no operand
		if c == '%' {
			p.add('%');	// TODO: should we bother with width & prec?
			continue;
		}
		if fieldnum >= v.Len() {	// out of operands
			p.add('%');
			p.add(c);
			p.addstr("(missing)");
			continue;
		}
		field := getField(v, fieldnum);
		fieldnum++;
		inter := field.Interface();
		if inter != nil && c != 'T' {	// don't want thing to describe itself if we're asking for its type
			if formatter, ok := inter.(Format); ok {
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
	if fieldnum < v.Len() {
		p.addstr("?(extra ");
		for ; fieldnum < v.Len(); fieldnum++ {
			p.addstr(getField(v, fieldnum).Type().String());
			if fieldnum + 1 < v.Len() {
				p.addstr(", ");
			}
		}
		p.addstr(")");
	}
}

func (p *pp) doprint(v reflect.StructValue, addspace, addnewline bool) {
	prev_string := false;
	for fieldnum := 0; fieldnum < v.Len();  fieldnum++ {
		// always add spaces if we're doing println
		field := getField(v, fieldnum);
		if fieldnum > 0 {
			if addspace {
				p.add(' ')
			} else if field.Kind() != reflect.StringKind && !prev_string{
				// if not doing println, add spaces if neither side is a string
				p.add(' ')
			}
		}
		was_string := p.printField(field);
		prev_string = was_string;
	}
	if addnewline {
		p.add('\n')
	}
}
