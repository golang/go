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
)

// Representation of printer state passed to custom formatters.
// Provides access to the io.Write interface plus information about
// the active formatting verb.
export type Formatter interface {
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

const Runeself = 0x80
const AllocSize = 32

type P struct {
	n	int;
	buf	[]byte;
	fmt	*Fmt;
}

func Printer() *P {
	p := new(*P);
	p.fmt = fmt.New();
	return p;
}

func (p *P) Width() (wid int, ok bool) {
	return p.fmt.wid, p.fmt.wid_present
}

func (p *P) Precision() (prec int, ok bool) {
	return p.fmt.prec, p.fmt.prec_present
}

func (p *P) Flag(b int) bool {
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

func (p *P) ensure(n int) {
	if len(p.buf) < n {
		newn := AllocSize + len(p.buf);
		if newn < n {
			newn = n + AllocSize
		}
		b := new([]byte, newn);
		for i := 0; i < p.n; i++ {
			b[i] = p.buf[i];
		}
		p.buf = b;
	}
}

func (p *P) addstr(s string) {
	n := len(s);
	p.ensure(p.n + n);
	for i := 0; i < n; i++ {
		p.buf[p.n] = s[i];
		p.n++;
	}
}

func (p *P) addbytes(b []byte, start, end int) {
	p.ensure(p.n + end-start);
	for i := start; i < end; i++ {
		p.buf[p.n] = b[i];
		p.n++;
	}
}

func (p *P) add(c int) {
	p.ensure(p.n + 1);
	if c < Runeself {
		p.buf[p.n] = byte(c);
		p.n++;
	} else {
		p.addstr(string(c));
	}
}

// Implement Write so we can call fprintf on a P, for
// recursive use in custom verbs.
func (p *P) Write(b []byte) (ret int, err *os.Error) {
	p.addbytes(b, 0, len(b));
	return len(b), nil;
}

func (p *P) doprintf(format string, v reflect.StructValue);
func (p *P) doprint(v reflect.StructValue, addspace, addnewline bool);

// These routines end in 'f' and take a format string.

export func fprintf(w io.Write, format string, a ...) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p := Printer();
	p.doprintf(format, v);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

export func printf(format string, v ...) (n int, errno *os.Error) {
	n, errno = fprintf(os.Stdout, format, v);
	return n, errno;
}

export func sprintf(format string, a ...) string {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p := Printer();
	p.doprintf(format, v);
	s := string(p.buf)[0 : p.n];
	return s;
}

// These routines do not take a format string and add spaces only
// when the operand on neither side is a string.

export func fprint(w io.Write, a ...) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p := Printer();
	p.doprint(v, false, false);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

export func print(v ...) (n int, errno *os.Error) {
	n, errno = fprint(os.Stdout, v);
	return n, errno;
}

export func sprint(a ...) string {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p := Printer();
	p.doprint(v, false, false);
	s := string(p.buf)[0 : p.n];
	return s;
}

// These routines end in 'ln', do not take a format string,
// always add spaces between operands, and add a newline
// after the last operand.

export func fprintln(w io.Write, a ...) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p := Printer();
	p.doprint(v, true, true);
	n, error = w.Write(p.buf[0:p.n]);
	return n, error;
}

export func println(v ...) (n int, errno *os.Error) {
	n, errno = fprintln(os.Stdout, v);
	return n, errno;
}

export func sprintln(a ...) string {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p := Printer();
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
	case reflect.PtrKind:
		if val, ok := v.Interface().(*[]byte); ok {
			return string(*val), true;
		}
	}
	// TODO(rsc): check for Interface().([]byte) too.
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

func (p *P) printField(field reflect.Value) (was_string bool) {
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
		s = p.fmt.boolean(field.(reflect.BoolValue).Get()).str();
	case reflect.IntKind, reflect.Int8Kind, reflect.Int16Kind, reflect.Int32Kind, reflect.Int64Kind:
		v, signed, ok := getInt(field);
		s = p.fmt.d64(v).str();
	case reflect.UintKind, reflect.Uint8Kind, reflect.Uint16Kind, reflect.Uint32Kind, reflect.Uint64Kind:
		v, signed, ok := getInt(field);
		s = p.fmt.ud64(uint64(v)).str();
	case reflect.UintptrKind:
		v, signed, ok := getInt(field);
		p.fmt.sharp = !p.fmt.sharp;  // turn 0x on by default
		s = p.fmt.ux64(uint64(v)).str();
	case reflect.Float32Kind:
		v, ok := getFloat32(field);
		s = p.fmt.g32(v).str();
	case reflect.Float64Kind, reflect.Float80Kind:
		v, ok := getFloat64(field);
		s = p.fmt.g64(v).str();
	case reflect.FloatKind:
		if field.Type().Size()*8 == 32 {
			v, ok := getFloat32(field);
			s = p.fmt.g32(v).str();
		} else {
			v, ok := getFloat64(field);
			s = p.fmt.g64(v).str();
		}
	case reflect.StringKind:
		v, ok := getString(field);
		s = p.fmt.s(v).str();
		was_string = true;
	case reflect.PtrKind:
		if v, ok := getPtr(field); v == 0 {
			s = "<nil>"
		} else {
			// pointer to array?
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
				s = p.fmt.uX64(uint64(v)).str();
			}
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

func (p *P) doprintf(format string, v reflect.StructValue) {
	p.ensure(len(format));	// a good starting size
	end := len(format) - 1;
	fieldnum := 0;	// we process one field per non-trivial format
	for i := 0; i <= end;  {
		c, w := sys.stringtorune(format, i);
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
		c, w = sys.stringtorune(format, i);
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
					s = p.fmt.b64(uint64(v)).str()	// always unsigned
				} else if v, ok := getFloat32(field); ok {
					s = p.fmt.fb32(v).str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.fb64(v).str()
				} else {
					goto badtype
				}
			case 'c':
				if v, signed, ok := getInt(field); ok {
					s = p.fmt.c(int(v)).str()
				} else {
					goto badtype
				}
			case 'd':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.d64(v).str()
					} else {
						s = p.fmt.ud64(uint64(v)).str()
					}
				} else {
					goto badtype
				}
			case 'o':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.o64(v).str()
					} else {
						s = p.fmt.uo64(uint64(v)).str()
					}
				} else {
					goto badtype
				}
			case 'x':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.x64(v).str()
					} else {
						s = p.fmt.ux64(uint64(v)).str()
					}
				} else if v, ok := getString(field); ok {
					s = p.fmt.sx(v).str();
				} else {
					goto badtype
				}
			case 'X':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.X64(v).str()
					} else {
						s = p.fmt.uX64(uint64(v)).str()
					}
				} else if v, ok := getString(field); ok {
					s = p.fmt.sX(v).str();
				} else {
					goto badtype
				}

			// float
			case 'e':
				if v, ok := getFloat32(field); ok {
					s = p.fmt.e32(v).str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.e64(v).str()
				} else {
					goto badtype
				}
			case 'f':
				if v, ok := getFloat32(field); ok {
					s = p.fmt.f32(v).str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.f64(v).str()
				} else {
					goto badtype
				}
			case 'g':
				if v, ok := getFloat32(field); ok {
					s = p.fmt.g32(v).str()
				} else if v, ok := getFloat64(field); ok {
					s = p.fmt.g64(v).str()
				} else {
					goto badtype
				}

			// string
			case 's':
				if v, ok := getString(field); ok {
					s = p.fmt.s(v).str()
				} else {
					goto badtype
				}
			case 'q':
				if v, ok := getString(field); ok {
					s = p.fmt.q(v).str()
				} else {
					goto badtype
				}

			// pointer
			case 'p':
				if v, ok := getPtr(field); ok {
					if v == 0 {
						s = "<nil>"
					} else {
						s = "0x" + p.fmt.uX64(uint64(v)).str()
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

func (p *P) doprint(v reflect.StructValue, addspace, addnewline bool) {
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
