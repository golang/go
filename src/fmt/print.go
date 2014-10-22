// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"errors"
	"io"
	"os"
	"reflect"
	"sync"
	"unicode/utf8"
)

// Some constants in the form of bytes, to avoid string overhead.
// Needlessly fastidious, I suppose.
var (
	commaSpaceBytes  = []byte(", ")
	nilAngleBytes    = []byte("<nil>")
	nilParenBytes    = []byte("(nil)")
	nilBytes         = []byte("nil")
	mapBytes         = []byte("map[")
	percentBangBytes = []byte("%!")
	missingBytes     = []byte("(MISSING)")
	badIndexBytes    = []byte("(BADINDEX)")
	panicBytes       = []byte("(PANIC=")
	extraBytes       = []byte("%!(EXTRA ")
	irparenBytes     = []byte("i)")
	bytesBytes       = []byte("[]byte{")
	badWidthBytes    = []byte("%!(BADWIDTH)")
	badPrecBytes     = []byte("%!(BADPREC)")
	noVerbBytes      = []byte("%!(NOVERB)")
)

// State represents the printer state passed to custom formatters.
// It provides access to the io.Writer interface plus information about
// the flags and options for the operand's format specifier.
type State interface {
	// Write is the function to call to emit formatted output to be printed.
	Write(b []byte) (ret int, err error)
	// Width returns the value of the width option and whether it has been set.
	Width() (wid int, ok bool)
	// Precision returns the value of the precision option and whether it has been set.
	Precision() (prec int, ok bool)

	// Flag reports whether the flag c, a character, has been set.
	Flag(c int) bool
}

// Formatter is the interface implemented by values with a custom formatter.
// The implementation of Format may call Sprint(f) or Fprint(f) etc.
// to generate its output.
type Formatter interface {
	Format(f State, c rune)
}

// Stringer is implemented by any value that has a String method,
// which defines the ``native'' format for that value.
// The String method is used to print values passed as an operand
// to any format that accepts a string or to an unformatted printer
// such as Print.
type Stringer interface {
	String() string
}

// GoStringer is implemented by any value that has a GoString method,
// which defines the Go syntax for that value.
// The GoString method is used to print values passed as an operand
// to a %#v format.
type GoStringer interface {
	GoString() string
}

// Use simple []byte instead of bytes.Buffer to avoid large dependency.
type buffer []byte

func (b *buffer) Write(p []byte) (n int, err error) {
	*b = append(*b, p...)
	return len(p), nil
}

func (b *buffer) WriteString(s string) (n int, err error) {
	*b = append(*b, s...)
	return len(s), nil
}

func (b *buffer) WriteByte(c byte) error {
	*b = append(*b, c)
	return nil
}

func (bp *buffer) WriteRune(r rune) error {
	if r < utf8.RuneSelf {
		*bp = append(*bp, byte(r))
		return nil
	}

	b := *bp
	n := len(b)
	for n+utf8.UTFMax > cap(b) {
		b = append(b, 0)
	}
	w := utf8.EncodeRune(b[n:n+utf8.UTFMax], r)
	*bp = b[:n+w]
	return nil
}

type pp struct {
	n         int
	panicking bool
	erroring  bool // printing an error condition
	buf       buffer
	// arg holds the current item, as an interface{}.
	arg interface{}
	// value holds the current item, as a reflect.Value, and will be
	// the zero Value if the item has not been reflected.
	value reflect.Value
	// reordered records whether the format string used argument reordering.
	reordered bool
	// goodArgNum records whether the most recent reordering directive was valid.
	goodArgNum bool
	runeBuf    [utf8.UTFMax]byte
	fmt        fmt
}

var ppFree = sync.Pool{
	New: func() interface{} { return new(pp) },
}

// newPrinter allocates a new pp struct or grabs a cached one.
func newPrinter() *pp {
	p := ppFree.Get().(*pp)
	p.panicking = false
	p.erroring = false
	p.fmt.init(&p.buf)
	return p
}

// free saves used pp structs in ppFree; avoids an allocation per invocation.
func (p *pp) free() {
	// Don't hold on to pp structs with large buffers.
	if cap(p.buf) > 1024 {
		return
	}
	p.buf = p.buf[:0]
	p.arg = nil
	p.value = reflect.Value{}
	ppFree.Put(p)
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

func (p *pp) add(c rune) {
	p.buf.WriteRune(c)
}

// Implement Write so we can call Fprintf on a pp (through State), for
// recursive use in custom verbs.
func (p *pp) Write(b []byte) (ret int, err error) {
	return p.buf.Write(b)
}

// These routines end in 'f' and take a format string.

// Fprintf formats according to a format specifier and writes to w.
// It returns the number of bytes written and any write error encountered.
func Fprintf(w io.Writer, format string, a ...interface{}) (n int, err error) {
	p := newPrinter()
	p.doPrintf(format, a)
	n, err = w.Write(p.buf)
	p.free()
	return
}

// Printf formats according to a format specifier and writes to standard output.
// It returns the number of bytes written and any write error encountered.
func Printf(format string, a ...interface{}) (n int, err error) {
	return Fprintf(os.Stdout, format, a...)
}

// Sprintf formats according to a format specifier and returns the resulting string.
func Sprintf(format string, a ...interface{}) string {
	p := newPrinter()
	p.doPrintf(format, a)
	s := string(p.buf)
	p.free()
	return s
}

// Errorf formats according to a format specifier and returns the string
// as a value that satisfies error.
func Errorf(format string, a ...interface{}) error {
	return errors.New(Sprintf(format, a...))
}

// These routines do not take a format string

// Fprint formats using the default formats for its operands and writes to w.
// Spaces are added between operands when neither is a string.
// It returns the number of bytes written and any write error encountered.
func Fprint(w io.Writer, a ...interface{}) (n int, err error) {
	p := newPrinter()
	p.doPrint(a, false, false)
	n, err = w.Write(p.buf)
	p.free()
	return
}

// Print formats using the default formats for its operands and writes to standard output.
// Spaces are added between operands when neither is a string.
// It returns the number of bytes written and any write error encountered.
func Print(a ...interface{}) (n int, err error) {
	return Fprint(os.Stdout, a...)
}

// Sprint formats using the default formats for its operands and returns the resulting string.
// Spaces are added between operands when neither is a string.
func Sprint(a ...interface{}) string {
	p := newPrinter()
	p.doPrint(a, false, false)
	s := string(p.buf)
	p.free()
	return s
}

// These routines end in 'ln', do not take a format string,
// always add spaces between operands, and add a newline
// after the last operand.

// Fprintln formats using the default formats for its operands and writes to w.
// Spaces are always added between operands and a newline is appended.
// It returns the number of bytes written and any write error encountered.
func Fprintln(w io.Writer, a ...interface{}) (n int, err error) {
	p := newPrinter()
	p.doPrint(a, true, true)
	n, err = w.Write(p.buf)
	p.free()
	return
}

// Println formats using the default formats for its operands and writes to standard output.
// Spaces are always added between operands and a newline is appended.
// It returns the number of bytes written and any write error encountered.
func Println(a ...interface{}) (n int, err error) {
	return Fprintln(os.Stdout, a...)
}

// Sprintln formats using the default formats for its operands and returns the resulting string.
// Spaces are always added between operands and a newline is appended.
func Sprintln(a ...interface{}) string {
	p := newPrinter()
	p.doPrint(a, true, true)
	s := string(p.buf)
	p.free()
	return s
}

// getField gets the i'th field of the struct value.
// If the field is itself is an interface, return a value for
// the thing inside the interface, not the interface itself.
func getField(v reflect.Value, i int) reflect.Value {
	val := v.Field(i)
	if val.Kind() == reflect.Interface && !val.IsNil() {
		val = val.Elem()
	}
	return val
}

// parsenum converts ASCII to integer.  num is 0 (and isnum is false) if no number present.
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

func (p *pp) unknownType(v reflect.Value) {
	if !v.IsValid() {
		p.buf.Write(nilAngleBytes)
		return
	}
	p.buf.WriteByte('?')
	p.buf.WriteString(v.Type().String())
	p.buf.WriteByte('?')
}

func (p *pp) badVerb(verb rune) {
	p.erroring = true
	p.add('%')
	p.add('!')
	p.add(verb)
	p.add('(')
	switch {
	case p.arg != nil:
		p.buf.WriteString(reflect.TypeOf(p.arg).String())
		p.add('=')
		p.printArg(p.arg, 'v', 0)
	case p.value.IsValid():
		p.buf.WriteString(p.value.Type().String())
		p.add('=')
		p.printValue(p.value, 'v', 0)
	default:
		p.buf.Write(nilAngleBytes)
	}
	p.add(')')
	p.erroring = false
}

func (p *pp) fmtBool(v bool, verb rune) {
	switch verb {
	case 't', 'v':
		p.fmt.fmt_boolean(v)
	default:
		p.badVerb(verb)
	}
}

// fmtC formats a rune for the 'c' format.
func (p *pp) fmtC(c int64) {
	r := rune(c) // Check for overflow.
	if int64(r) != c {
		r = utf8.RuneError
	}
	w := utf8.EncodeRune(p.runeBuf[0:utf8.UTFMax], r)
	p.fmt.pad(p.runeBuf[0:w])
}

func (p *pp) fmtInt64(v int64, verb rune) {
	switch verb {
	case 'b':
		p.fmt.integer(v, 2, signed, ldigits)
	case 'c':
		p.fmtC(v)
	case 'd', 'v':
		p.fmt.integer(v, 10, signed, ldigits)
	case 'o':
		p.fmt.integer(v, 8, signed, ldigits)
	case 'q':
		if 0 <= v && v <= utf8.MaxRune {
			p.fmt.fmt_qc(v)
		} else {
			p.badVerb(verb)
		}
	case 'x':
		p.fmt.integer(v, 16, signed, ldigits)
	case 'U':
		p.fmtUnicode(v)
	case 'X':
		p.fmt.integer(v, 16, signed, udigits)
	default:
		p.badVerb(verb)
	}
}

// fmt0x64 formats a uint64 in hexadecimal and prefixes it with 0x or
// not, as requested, by temporarily setting the sharp flag.
func (p *pp) fmt0x64(v uint64, leading0x bool) {
	sharp := p.fmt.sharp
	p.fmt.sharp = leading0x
	p.fmt.integer(int64(v), 16, unsigned, ldigits)
	p.fmt.sharp = sharp
}

// fmtUnicode formats a uint64 in U+1234 form by
// temporarily turning on the unicode flag and tweaking the precision.
func (p *pp) fmtUnicode(v int64) {
	precPresent := p.fmt.precPresent
	sharp := p.fmt.sharp
	p.fmt.sharp = false
	prec := p.fmt.prec
	if !precPresent {
		// If prec is already set, leave it alone; otherwise 4 is minimum.
		p.fmt.prec = 4
		p.fmt.precPresent = true
	}
	p.fmt.unicode = true // turn on U+
	p.fmt.uniQuote = sharp
	p.fmt.integer(int64(v), 16, unsigned, udigits)
	p.fmt.unicode = false
	p.fmt.uniQuote = false
	p.fmt.prec = prec
	p.fmt.precPresent = precPresent
	p.fmt.sharp = sharp
}

func (p *pp) fmtUint64(v uint64, verb rune) {
	switch verb {
	case 'b':
		p.fmt.integer(int64(v), 2, unsigned, ldigits)
	case 'c':
		p.fmtC(int64(v))
	case 'd':
		p.fmt.integer(int64(v), 10, unsigned, ldigits)
	case 'v':
		if p.fmt.sharpV {
			p.fmt0x64(v, true)
		} else {
			p.fmt.integer(int64(v), 10, unsigned, ldigits)
		}
	case 'o':
		p.fmt.integer(int64(v), 8, unsigned, ldigits)
	case 'q':
		if 0 <= v && v <= utf8.MaxRune {
			p.fmt.fmt_qc(int64(v))
		} else {
			p.badVerb(verb)
		}
	case 'x':
		p.fmt.integer(int64(v), 16, unsigned, ldigits)
	case 'X':
		p.fmt.integer(int64(v), 16, unsigned, udigits)
	case 'U':
		p.fmtUnicode(int64(v))
	default:
		p.badVerb(verb)
	}
}

func (p *pp) fmtFloat32(v float32, verb rune) {
	switch verb {
	case 'b':
		p.fmt.fmt_fb32(v)
	case 'e':
		p.fmt.fmt_e32(v)
	case 'E':
		p.fmt.fmt_E32(v)
	case 'f', 'F':
		p.fmt.fmt_f32(v)
	case 'g', 'v':
		p.fmt.fmt_g32(v)
	case 'G':
		p.fmt.fmt_G32(v)
	default:
		p.badVerb(verb)
	}
}

func (p *pp) fmtFloat64(v float64, verb rune) {
	switch verb {
	case 'b':
		p.fmt.fmt_fb64(v)
	case 'e':
		p.fmt.fmt_e64(v)
	case 'E':
		p.fmt.fmt_E64(v)
	case 'f', 'F':
		p.fmt.fmt_f64(v)
	case 'g', 'v':
		p.fmt.fmt_g64(v)
	case 'G':
		p.fmt.fmt_G64(v)
	default:
		p.badVerb(verb)
	}
}

func (p *pp) fmtComplex64(v complex64, verb rune) {
	switch verb {
	case 'b', 'e', 'E', 'f', 'F', 'g', 'G':
		p.fmt.fmt_c64(v, verb)
	case 'v':
		p.fmt.fmt_c64(v, 'g')
	default:
		p.badVerb(verb)
	}
}

func (p *pp) fmtComplex128(v complex128, verb rune) {
	switch verb {
	case 'b', 'e', 'E', 'f', 'F', 'g', 'G':
		p.fmt.fmt_c128(v, verb)
	case 'v':
		p.fmt.fmt_c128(v, 'g')
	default:
		p.badVerb(verb)
	}
}

func (p *pp) fmtString(v string, verb rune) {
	switch verb {
	case 'v':
		if p.fmt.sharpV {
			p.fmt.fmt_q(v)
		} else {
			p.fmt.fmt_s(v)
		}
	case 's':
		p.fmt.fmt_s(v)
	case 'x':
		p.fmt.fmt_sx(v, ldigits)
	case 'X':
		p.fmt.fmt_sx(v, udigits)
	case 'q':
		p.fmt.fmt_q(v)
	default:
		p.badVerb(verb)
	}
}

func (p *pp) fmtBytes(v []byte, verb rune, typ reflect.Type, depth int) {
	if verb == 'v' || verb == 'd' {
		if p.fmt.sharpV {
			if v == nil {
				if typ == nil {
					p.buf.WriteString("[]byte(nil)")
				} else {
					p.buf.WriteString(typ.String())
					p.buf.Write(nilParenBytes)
				}
				return
			}
			if typ == nil {
				p.buf.Write(bytesBytes)
			} else {
				p.buf.WriteString(typ.String())
				p.buf.WriteByte('{')
			}
		} else {
			p.buf.WriteByte('[')
		}
		for i, c := range v {
			if i > 0 {
				if p.fmt.sharpV {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			p.printArg(c, 'v', depth+1)
		}
		if p.fmt.sharpV {
			p.buf.WriteByte('}')
		} else {
			p.buf.WriteByte(']')
		}
		return
	}
	switch verb {
	case 's':
		p.fmt.fmt_s(string(v))
	case 'x':
		p.fmt.fmt_bx(v, ldigits)
	case 'X':
		p.fmt.fmt_bx(v, udigits)
	case 'q':
		p.fmt.fmt_q(string(v))
	default:
		p.badVerb(verb)
	}
}

func (p *pp) fmtPointer(value reflect.Value, verb rune) {
	use0x64 := true
	switch verb {
	case 'p', 'v':
		// ok
	case 'b', 'd', 'o', 'x', 'X':
		use0x64 = false
		// ok
	default:
		p.badVerb(verb)
		return
	}

	var u uintptr
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Map, reflect.Ptr, reflect.Slice, reflect.UnsafePointer:
		u = value.Pointer()
	default:
		p.badVerb(verb)
		return
	}

	if p.fmt.sharpV {
		p.add('(')
		p.buf.WriteString(value.Type().String())
		p.add(')')
		p.add('(')
		if u == 0 {
			p.buf.Write(nilBytes)
		} else {
			p.fmt0x64(uint64(u), true)
		}
		p.add(')')
	} else if verb == 'v' && u == 0 {
		p.buf.Write(nilAngleBytes)
	} else {
		if use0x64 {
			p.fmt0x64(uint64(u), !p.fmt.sharp)
		} else {
			p.fmtUint64(uint64(u), verb)
		}
	}
}

var (
	intBits     = reflect.TypeOf(0).Bits()
	uintptrBits = reflect.TypeOf(uintptr(0)).Bits()
)

func (p *pp) catchPanic(arg interface{}, verb rune) {
	if err := recover(); err != nil {
		// If it's a nil pointer, just say "<nil>". The likeliest causes are a
		// Stringer that fails to guard against nil or a nil pointer for a
		// value receiver, and in either case, "<nil>" is a nice result.
		if v := reflect.ValueOf(arg); v.Kind() == reflect.Ptr && v.IsNil() {
			p.buf.Write(nilAngleBytes)
			return
		}
		// Otherwise print a concise panic message. Most of the time the panic
		// value will print itself nicely.
		if p.panicking {
			// Nested panics; the recursion in printArg cannot succeed.
			panic(err)
		}
		p.fmt.clearflags() // We are done, and for this output we want default behavior.
		p.buf.Write(percentBangBytes)
		p.add(verb)
		p.buf.Write(panicBytes)
		p.panicking = true
		p.printArg(err, 'v', 0)
		p.panicking = false
		p.buf.WriteByte(')')
	}
}

// clearSpecialFlags pushes %#v back into the regular flags and returns their old state.
func (p *pp) clearSpecialFlags() (plusV, sharpV bool) {
	plusV = p.fmt.plusV
	if plusV {
		p.fmt.plus = true
		p.fmt.plusV = false
	}
	sharpV = p.fmt.sharpV
	if sharpV {
		p.fmt.sharp = true
		p.fmt.sharpV = false
	}
	return
}

// restoreSpecialFlags, whose argument should be a call to clearSpecialFlags,
// restores the setting of the plusV and sharpV flags.
func (p *pp) restoreSpecialFlags(plusV, sharpV bool) {
	if plusV {
		p.fmt.plus = false
		p.fmt.plusV = true
	}
	if sharpV {
		p.fmt.sharp = false
		p.fmt.sharpV = true
	}
}

func (p *pp) handleMethods(verb rune, depth int) (handled bool) {
	if p.erroring {
		return
	}
	// Is it a Formatter?
	if formatter, ok := p.arg.(Formatter); ok {
		handled = true
		defer p.restoreSpecialFlags(p.clearSpecialFlags())
		defer p.catchPanic(p.arg, verb)
		formatter.Format(p, verb)
		return
	}

	// If we're doing Go syntax and the argument knows how to supply it, take care of it now.
	if p.fmt.sharpV {
		if stringer, ok := p.arg.(GoStringer); ok {
			handled = true
			defer p.catchPanic(p.arg, verb)
			// Print the result of GoString unadorned.
			p.fmt.fmt_s(stringer.GoString())
			return
		}
	} else {
		// If a string is acceptable according to the format, see if
		// the value satisfies one of the string-valued interfaces.
		// Println etc. set verb to %v, which is "stringable".
		switch verb {
		case 'v', 's', 'x', 'X', 'q':
			// Is it an error or Stringer?
			// The duplication in the bodies is necessary:
			// setting handled and deferring catchPanic
			// must happen before calling the method.
			switch v := p.arg.(type) {
			case error:
				handled = true
				defer p.catchPanic(p.arg, verb)
				p.printArg(v.Error(), verb, depth)
				return

			case Stringer:
				handled = true
				defer p.catchPanic(p.arg, verb)
				p.printArg(v.String(), verb, depth)
				return
			}
		}
	}
	return false
}

func (p *pp) printArg(arg interface{}, verb rune, depth int) (wasString bool) {
	p.arg = arg
	p.value = reflect.Value{}

	if arg == nil {
		if verb == 'T' || verb == 'v' {
			p.fmt.pad(nilAngleBytes)
		} else {
			p.badVerb(verb)
		}
		return false
	}

	// Special processing considerations.
	// %T (the value's type) and %p (its address) are special; we always do them first.
	switch verb {
	case 'T':
		p.printArg(reflect.TypeOf(arg).String(), 's', 0)
		return false
	case 'p':
		p.fmtPointer(reflect.ValueOf(arg), verb)
		return false
	}

	// Some types can be done without reflection.
	switch f := arg.(type) {
	case bool:
		p.fmtBool(f, verb)
	case float32:
		p.fmtFloat32(f, verb)
	case float64:
		p.fmtFloat64(f, verb)
	case complex64:
		p.fmtComplex64(f, verb)
	case complex128:
		p.fmtComplex128(f, verb)
	case int:
		p.fmtInt64(int64(f), verb)
	case int8:
		p.fmtInt64(int64(f), verb)
	case int16:
		p.fmtInt64(int64(f), verb)
	case int32:
		p.fmtInt64(int64(f), verb)
	case int64:
		p.fmtInt64(f, verb)
	case uint:
		p.fmtUint64(uint64(f), verb)
	case uint8:
		p.fmtUint64(uint64(f), verb)
	case uint16:
		p.fmtUint64(uint64(f), verb)
	case uint32:
		p.fmtUint64(uint64(f), verb)
	case uint64:
		p.fmtUint64(f, verb)
	case uintptr:
		p.fmtUint64(uint64(f), verb)
	case string:
		p.fmtString(f, verb)
		wasString = verb == 's' || verb == 'v'
	case []byte:
		p.fmtBytes(f, verb, nil, depth)
		wasString = verb == 's'
	default:
		// If the type is not simple, it might have methods.
		if handled := p.handleMethods(verb, depth); handled {
			return false
		}
		// Need to use reflection
		return p.printReflectValue(reflect.ValueOf(arg), verb, depth)
	}
	p.arg = nil
	return
}

// printValue is like printArg but starts with a reflect value, not an interface{} value.
func (p *pp) printValue(value reflect.Value, verb rune, depth int) (wasString bool) {
	if !value.IsValid() {
		if verb == 'T' || verb == 'v' {
			p.buf.Write(nilAngleBytes)
		} else {
			p.badVerb(verb)
		}
		return false
	}

	// Special processing considerations.
	// %T (the value's type) and %p (its address) are special; we always do them first.
	switch verb {
	case 'T':
		p.printArg(value.Type().String(), 's', 0)
		return false
	case 'p':
		p.fmtPointer(value, verb)
		return false
	}

	// Handle values with special methods.
	// Call always, even when arg == nil, because handleMethods clears p.fmt.plus for us.
	p.arg = nil // Make sure it's cleared, for safety.
	if value.CanInterface() {
		p.arg = value.Interface()
	}
	if handled := p.handleMethods(verb, depth); handled {
		return false
	}

	return p.printReflectValue(value, verb, depth)
}

var byteType = reflect.TypeOf(byte(0))

// printReflectValue is the fallback for both printArg and printValue.
// It uses reflect to print the value.
func (p *pp) printReflectValue(value reflect.Value, verb rune, depth int) (wasString bool) {
	oldValue := p.value
	p.value = value
BigSwitch:
	switch f := value; f.Kind() {
	case reflect.Bool:
		p.fmtBool(f.Bool(), verb)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		p.fmtInt64(f.Int(), verb)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		p.fmtUint64(f.Uint(), verb)
	case reflect.Float32, reflect.Float64:
		if f.Type().Size() == 4 {
			p.fmtFloat32(float32(f.Float()), verb)
		} else {
			p.fmtFloat64(f.Float(), verb)
		}
	case reflect.Complex64, reflect.Complex128:
		if f.Type().Size() == 8 {
			p.fmtComplex64(complex64(f.Complex()), verb)
		} else {
			p.fmtComplex128(f.Complex(), verb)
		}
	case reflect.String:
		p.fmtString(f.String(), verb)
	case reflect.Map:
		if p.fmt.sharpV {
			p.buf.WriteString(f.Type().String())
			if f.IsNil() {
				p.buf.WriteString("(nil)")
				break
			}
			p.buf.WriteByte('{')
		} else {
			p.buf.Write(mapBytes)
		}
		keys := f.MapKeys()
		for i, key := range keys {
			if i > 0 {
				if p.fmt.sharpV {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			p.printValue(key, verb, depth+1)
			p.buf.WriteByte(':')
			p.printValue(f.MapIndex(key), verb, depth+1)
		}
		if p.fmt.sharpV {
			p.buf.WriteByte('}')
		} else {
			p.buf.WriteByte(']')
		}
	case reflect.Struct:
		if p.fmt.sharpV {
			p.buf.WriteString(value.Type().String())
		}
		p.add('{')
		v := f
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			if i > 0 {
				if p.fmt.sharpV {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			if p.fmt.plusV || p.fmt.sharpV {
				if f := t.Field(i); f.Name != "" {
					p.buf.WriteString(f.Name)
					p.buf.WriteByte(':')
				}
			}
			p.printValue(getField(v, i), verb, depth+1)
		}
		p.buf.WriteByte('}')
	case reflect.Interface:
		value := f.Elem()
		if !value.IsValid() {
			if p.fmt.sharpV {
				p.buf.WriteString(f.Type().String())
				p.buf.Write(nilParenBytes)
			} else {
				p.buf.Write(nilAngleBytes)
			}
		} else {
			wasString = p.printValue(value, verb, depth+1)
		}
	case reflect.Array, reflect.Slice:
		// Byte slices are special:
		// - Handle []byte (== []uint8) with fmtBytes.
		// - Handle []T, where T is a named byte type, with fmtBytes only
		//   for the s, q, an x verbs. For other verbs, T might be a
		//   Stringer, so we use printValue to print each element.
		if typ := f.Type(); typ.Elem().Kind() == reflect.Uint8 && (typ.Elem() == byteType || verb == 's' || verb == 'q' || verb == 'x') {
			var bytes []byte
			if f.Kind() == reflect.Slice {
				bytes = f.Bytes()
			} else if f.CanAddr() {
				bytes = f.Slice(0, f.Len()).Bytes()
			} else {
				// We have an array, but we cannot Slice() a non-addressable array,
				// so we build a slice by hand. This is a rare case but it would be nice
				// if reflection could help a little more.
				bytes = make([]byte, f.Len())
				for i := range bytes {
					bytes[i] = byte(f.Index(i).Uint())
				}
			}
			p.fmtBytes(bytes, verb, typ, depth)
			wasString = verb == 's'
			break
		}
		if p.fmt.sharpV {
			p.buf.WriteString(value.Type().String())
			if f.Kind() == reflect.Slice && f.IsNil() {
				p.buf.WriteString("(nil)")
				break
			}
			p.buf.WriteByte('{')
		} else {
			p.buf.WriteByte('[')
		}
		for i := 0; i < f.Len(); i++ {
			if i > 0 {
				if p.fmt.sharpV {
					p.buf.Write(commaSpaceBytes)
				} else {
					p.buf.WriteByte(' ')
				}
			}
			p.printValue(f.Index(i), verb, depth+1)
		}
		if p.fmt.sharpV {
			p.buf.WriteByte('}')
		} else {
			p.buf.WriteByte(']')
		}
	case reflect.Ptr:
		v := f.Pointer()
		// pointer to array or slice or struct?  ok at top level
		// but not embedded (avoid loops)
		if v != 0 && depth == 0 {
			switch a := f.Elem(); a.Kind() {
			case reflect.Array, reflect.Slice:
				p.buf.WriteByte('&')
				p.printValue(a, verb, depth+1)
				break BigSwitch
			case reflect.Struct:
				p.buf.WriteByte('&')
				p.printValue(a, verb, depth+1)
				break BigSwitch
			case reflect.Map:
				p.buf.WriteByte('&')
				p.printValue(a, verb, depth+1)
				break BigSwitch
			}
		}
		fallthrough
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		p.fmtPointer(value, verb)
	default:
		p.unknownType(f)
	}
	p.value = oldValue
	return wasString
}

// intFromArg gets the argNumth element of a. On return, isInt reports whether the argument has type int.
func intFromArg(a []interface{}, argNum int) (num int, isInt bool, newArgNum int) {
	newArgNum = argNum
	if argNum < len(a) {
		num, isInt = a[argNum].(int)
		newArgNum = argNum + 1
	}
	return
}

// parseArgNumber returns the value of the bracketed number, minus 1
// (explicit argument numbers are one-indexed but we want zero-indexed).
// The opening bracket is known to be present at format[0].
// The returned values are the index, the number of bytes to consume
// up to the closing paren, if present, and whether the number parsed
// ok. The bytes to consume will be 1 if no closing paren is present.
func parseArgNumber(format string) (index int, wid int, ok bool) {
	// Find closing bracket.
	for i := 1; i < len(format); i++ {
		if format[i] == ']' {
			width, ok, newi := parsenum(format, 1, i)
			if !ok || newi != i {
				return 0, i + 1, false
			}
			return width - 1, i + 1, true // arg numbers are one-indexed and skip paren.
		}
	}
	return 0, 1, false
}

// argNumber returns the next argument to evaluate, which is either the value of the passed-in
// argNum or the value of the bracketed integer that begins format[i:]. It also returns
// the new value of i, that is, the index of the next byte of the format to process.
func (p *pp) argNumber(argNum int, format string, i int, numArgs int) (newArgNum, newi int, found bool) {
	if len(format) <= i || format[i] != '[' {
		return argNum, i, false
	}
	p.reordered = true
	index, wid, ok := parseArgNumber(format[i:])
	if ok && 0 <= index && index < numArgs {
		return index, i + wid, true
	}
	p.goodArgNum = false
	return argNum, i + wid, true
}

func (p *pp) doPrintf(format string, a []interface{}) {
	end := len(format)
	argNum := 0         // we process one argument per non-trivial format
	afterIndex := false // previous item in format was an index like [3].
	p.reordered = false
	for i := 0; i < end; {
		p.goodArgNum = true
		lasti := i
		for i < end && format[i] != '%' {
			i++
		}
		if i > lasti {
			p.buf.WriteString(format[lasti:i])
		}
		if i >= end {
			// done processing format string
			break
		}

		// Process one verb
		i++

		// Do we have flags?
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

		// Do we have an explicit argument index?
		argNum, i, afterIndex = p.argNumber(argNum, format, i, len(a))

		// Do we have width?
		if i < end && format[i] == '*' {
			i++
			p.fmt.wid, p.fmt.widPresent, argNum = intFromArg(a, argNum)
			if !p.fmt.widPresent {
				p.buf.Write(badWidthBytes)
			}
			afterIndex = false
		} else {
			p.fmt.wid, p.fmt.widPresent, i = parsenum(format, i, end)
			if afterIndex && p.fmt.widPresent { // "%[3]2d"
				p.goodArgNum = false
			}
		}

		// Do we have precision?
		if i+1 < end && format[i] == '.' {
			i++
			if afterIndex { // "%[3].2d"
				p.goodArgNum = false
			}
			argNum, i, afterIndex = p.argNumber(argNum, format, i, len(a))
			if format[i] == '*' {
				i++
				p.fmt.prec, p.fmt.precPresent, argNum = intFromArg(a, argNum)
				if !p.fmt.precPresent {
					p.buf.Write(badPrecBytes)
				}
				afterIndex = false
			} else {
				p.fmt.prec, p.fmt.precPresent, i = parsenum(format, i, end)
				if !p.fmt.precPresent {
					p.fmt.prec = 0
					p.fmt.precPresent = true
				}
			}
		}

		if !afterIndex {
			argNum, i, afterIndex = p.argNumber(argNum, format, i, len(a))
		}

		if i >= end {
			p.buf.Write(noVerbBytes)
			continue
		}
		c, w := utf8.DecodeRuneInString(format[i:])
		i += w
		// percent is special - absorbs no operand
		if c == '%' {
			p.buf.WriteByte('%') // We ignore width and prec.
			continue
		}
		if !p.goodArgNum {
			p.buf.Write(percentBangBytes)
			p.add(c)
			p.buf.Write(badIndexBytes)
			continue
		} else if argNum >= len(a) { // out of operands
			p.buf.Write(percentBangBytes)
			p.add(c)
			p.buf.Write(missingBytes)
			continue
		}
		arg := a[argNum]
		argNum++

		if c == 'v' {
			if p.fmt.sharp {
				// Go syntax. Set the flag in the fmt and clear the sharp flag.
				p.fmt.sharp = false
				p.fmt.sharpV = true
			}
			if p.fmt.plus {
				// Struct-field syntax. Set the flag in the fmt and clear the plus flag.
				p.fmt.plus = false
				p.fmt.plusV = true
			}
		}
		p.printArg(arg, c, 0)
	}

	// Check for extra arguments unless the call accessed the arguments
	// out of order, in which case it's too expensive to detect if they've all
	// been used and arguably OK if they're not.
	if !p.reordered && argNum < len(a) {
		p.buf.Write(extraBytes)
		for ; argNum < len(a); argNum++ {
			arg := a[argNum]
			if arg != nil {
				p.buf.WriteString(reflect.TypeOf(arg).String())
				p.buf.WriteByte('=')
			}
			p.printArg(arg, 'v', 0)
			if argNum+1 < len(a) {
				p.buf.Write(commaSpaceBytes)
			}
		}
		p.buf.WriteByte(')')
	}
}

func (p *pp) doPrint(a []interface{}, addspace, addnewline bool) {
	prevString := false
	for argNum := 0; argNum < len(a); argNum++ {
		p.fmt.clearflags()
		// always add spaces if we're doing Println
		arg := a[argNum]
		if argNum > 0 {
			isString := arg != nil && reflect.TypeOf(arg).Kind() == reflect.String
			if addspace || !isString && !prevString {
				p.buf.WriteByte(' ')
			}
		}
		prevString = p.printArg(arg, 'v', 0)
	}
	if addnewline {
		p.buf.WriteByte('\n')
	}
}
