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
	"reflect";
	"os";
)

const Runeself = 0x80
const AllocSize = 32

export type P struct {
	n	int;
	buf	*[]byte;
	fmt	*Fmt;
}

export func Printer() *P {
	p := new(P);
	p.fmt = fmt.New();
	return p;
}

func (p *P) ensure(n int) {
	if p.buf == nil || len(p.buf) < n {
		newn := AllocSize;
		if p.buf != nil {
			newn += len(p.buf);
		}
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

func (p *P) addbytes(b *[]byte, start, end int) {
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

func (p *P) reset() {
	p.n = 0;
}

export type Writer interface {
	Write(b *[]byte) (ret int, err *os.Error);
}

func (p *P) doprintf(format string, v reflect.StructValue);
func (p *P) doprint(v reflect.StructValue, addspace bool);

// These routines end in 'f' and take a format string.

func (p *P) fprintf(w Writer, format string, a reflect.Empty) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p.doprintf(format, v);
	n, error = w.Write(p.buf[0:p.n]);
	p.reset();
	return n, error;
}

func (p *P) printf(format string, v reflect.Empty) (n int, errno *os.Error) {
	n, errno = p.fprintf(os.Stdout, format, v);
	return n, errno;
}

func (p *P) sprintf(format string, v reflect.Empty) string {
	p.doprintf(format, reflect.NewValue(v).(reflect.StructValue));
	s := string(p.buf)[0 : p.n];
	p.reset();
	return s;
}

// These routines do not take a format string and add spaces only
// when the operand on neither side is a string.

func (p *P) fprint(w Writer, a reflect.Empty) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p.doprint(v, false);
	n, error = w.Write(p.buf[0:p.n]);
	p.reset();
	return n, error;
}

func (p *P) print(v reflect.Empty) (n int, errno *os.Error) {
	n, errno = p.fprint(os.Stdout, v);
	return n, errno;
}

func (p *P) sprint(v reflect.Empty) string {
	p.doprint(reflect.NewValue(v).(reflect.StructValue), false);
	s := string(p.buf)[0 : p.n];
	p.reset();
	return s;
}

// These routines end in 'ln', do not take a format string,
// always add spaces between operands, and add a newline
// after the last operand.

func (p *P) fprintln(w Writer, a reflect.Empty) (n int, error *os.Error) {
	v := reflect.NewValue(a).(reflect.PtrValue).Sub().(reflect.StructValue);
	p.doprint(v, true);
	n, error = w.Write(p.buf[0:p.n]);
	p.reset();
	return n, error;
}

func (p *P) println(v reflect.Empty) (n int, errno *os.Error) {
	n, errno = p.fprintln(os.Stdout, v);
	return n, errno;
}

func (p *P) sprintln(v reflect.Empty) string {
	p.doprint(reflect.NewValue(v).(reflect.StructValue), true);
	s := string(p.buf)[0 : p.n];
	p.reset();
	return s;
}

// Getters for the fields of the argument structure.

func getInt(v reflect.Value) (val int64, signed, ok bool) {
	switch v.Kind() {
	case reflect.Int8Kind:
		return int64(v.(reflect.Int8Value).Get()), true, true;
	case reflect.Int16Kind:
		return int64(v.(reflect.Int16Value).Get()), true, true;
	case reflect.Int32Kind:
		return int64(v.(reflect.Int32Value).Get()), true, true;
	case reflect.Int64Kind:
		return int64(v.(reflect.Int64Value).Get()), true, true;
	case reflect.Uint8Kind:
		return int64(v.(reflect.Uint8Value).Get()), false, true;
	case reflect.Uint16Kind:
		return int64(v.(reflect.Uint16Value).Get()), false, true;
	case reflect.Uint32Kind:
		return int64(v.(reflect.Uint32Value).Get()), false, true;
	case reflect.Uint64Kind:
		return int64(v.(reflect.Uint64Value).Get()), false, true;
	}
	return 0, false, false;
}

func getString(v reflect.Value) (val string, ok bool) {
	switch v.Kind() {
	case reflect.StringKind:
		return v.(reflect.StringValue).Get(), true;
	}
	return "", false;
}

func getFloat(v reflect.Value) (val float64, ok bool) {
	switch v.Kind() {
	case reflect.Float32Kind:
		return float64(v.(reflect.Float32Value).Get()), true;
	case reflect.Float64Kind:
		return float64(v.(reflect.Float32Value).Get()), true;
	case reflect.Float80Kind:
		break;	// TODO: what to do here?
	}
	return 0.0, false;
}

func getPtr(v reflect.Value) (val uint64, ok bool) {
	switch v.Kind() {
	case reflect.PtrKind:
		return v.(reflect.PtrValue).Get(), true;
	}
	return 0, false;
}

// Convert ASCII to integer.

func parsenum(s string, start, end int) (n int, got bool, newi int) {
	if start >= end {
		return 0, false, end
	}
	if s[start] == '-' {
		a, b, c := parsenum(s, start+1, end);
		if b {
			return -a, b, c;
		}
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
		var got bool;
		// saw % - do we have %20 (width)?
		w, got, i = parsenum(format, i+1, end);
		if got {
			p.fmt.w(w);
		}
		// do we have %.20 (precision)?
		if i < end && format[i] == '.' {
			w, got, i = parsenum(format, i+1, end);
			if got {
				p.fmt.p(w);
			}
		}
		c, w = sys.stringtorune(format, i);
		i += w;
		// percent is special - absorbs no operand
		if c == '%' {
			p.add('%');	// TODO: should we bother with width & prec?
			continue;
		}
		if fieldnum >= v.Len() {	// out of operands
			p.addstr("???");
			continue;
		}
		field := v.Field(fieldnum);
		fieldnum++;
		s := "";
		switch c {
			// int
			case 'b':
				if v, signed, ok := getInt(field); ok {
					s = p.fmt.b64(uint64(v)).str()	// always unsigned
				} else {
					s = "%b%"
				}
			case 'd':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.d64(v).str()
					} else {
						s = p.fmt.ud64(uint64(v)).str()
					}
				} else {
					s = "%d%"
				}
			case 'o':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.o64(v).str()
					} else {
						s = p.fmt.uo64(uint64(v)).str()
					}
				} else {
					s= "%o%"
				}
			case 'x':
				if v, signed, ok := getInt(field); ok {
					if signed {
						s = p.fmt.x64(v).str()
					} else {
						s = p.fmt.ux64(uint64(v)).str()
					}
				} else {
					s = "%x%"
				}

			// float
			case 'e':
				if v, ok := getFloat(field); ok {
					s = p.fmt.e64(v).str()
				} else {
					s = "%e%"
				}
			case 'f':
				if v, ok := getFloat(field); ok {
					s = p.fmt.f64(v).str()
				} else {
					s = "%f%";
				}
			case 'g':
				if v, ok := getFloat(field); ok {
					s = p.fmt.g64(v).str()
				} else {
					s = "%g%"
				}

			// string
			case 's':
				if v, ok := getString(field); ok {
					s = p.fmt.s(v).str()
				} else {
					s = "%s%"
				}

			// pointer
			case 'p':
				if v, ok := getPtr(field); ok {
					s = "0x" + p.fmt.uX64(v).str()
				} else {
					s = "%p%"
				}

			default:
				s = "?" + string(c) + "?";
		}
		p.addstr(s);
	}
}

func (p *P) doprint(v reflect.StructValue, is_println bool) {
	prev_string := false;
	for fieldnum := 0; fieldnum < v.Len();  fieldnum++ {
		// always add spaces if we're doing println
		field := v.Field(fieldnum);
		s := "";
		if is_println {
			if fieldnum > 0 {
				p.add(' ')
			}
		} else if field.Kind() != reflect.StringKind && !prev_string{
			// if not doing println, add spaces if neither side is a string
			p.add(' ')
		}
		switch field.Kind() {
		case reflect.Int8Kind, reflect.Int16Kind, reflect.Int32Kind, reflect.Int64Kind:
			v, signed, ok := getInt(field);
			s = p.fmt.d64(v).str();
		case reflect.Uint8Kind, reflect.Uint16Kind, reflect.Uint32Kind, reflect.Uint64Kind:
			v, signed, ok := getInt(field);
			s = p.fmt.ud64(uint64(v)).str();
		case reflect.Float32Kind, reflect.Float64Kind, reflect.Float80Kind:
			v, ok := getFloat(field);
			s = p.fmt.g64(v).str();
		case reflect.StringKind:
			v, ok := getString(field);
			s = p.fmt.s(v).str();
		case reflect.PtrKind:
			v, ok := getPtr(field);
			p.add('0');
			p.add('x');
			s = p.fmt.uX64(v).str();
		default:
			s = "???";
		}
		p.addstr(s);
		prev_string = field.Kind() == reflect.StringKind;
	}
	if is_println {
		p.add('\n')
	}
}
