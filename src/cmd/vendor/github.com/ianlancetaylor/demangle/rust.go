// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package demangle

import (
	"fmt"
	"math"
	"math/bits"
	"strings"
	"unicode/utf8"
)

// rustToString demangles a Rust symbol.
func rustToString(name string, options []Option) (ret string, err error) {
	if !strings.HasPrefix(name, "_R") {
		return "", ErrNotMangledName
	}

	// When the demangling routines encounter an error, they panic
	// with a value of type demangleErr.
	defer func() {
		if r := recover(); r != nil {
			if de, ok := r.(demangleErr); ok {
				ret = ""
				err = de
				return
			}
			panic(r)
		}
	}()

	suffix := ""
	dot := strings.Index(name, ".")
	if dot >= 0 {
		suffix = name[dot:]
		name = name[:dot]
	}

	name = name[2:]
	rst := &rustState{orig: name, str: name}
	rst.symbolName()

	if len(rst.str) > 0 {
		rst.fail("unparsed characters at end of mangled name")
	}

	if suffix != "" {
		rst.skip = false
		rst.writeString(" (")
		rst.writeString(suffix)
		rst.writeByte(')')
	}

	return rst.buf.String(), nil
}

// A rustState holds the current state of demangling a Rust string.
type rustState struct {
	orig      string          // the original string being demangled
	str       string          // remainder of string to demangle
	off       int             // offset of str within original string
	buf       strings.Builder // demangled string being built
	skip      bool            // don't print, just skip
	lifetimes int64           // number of bound lifetimes
	last      byte            // last byte written to buffer
}

// fail panics with demangleErr, to be caught in rustToString.
func (rst *rustState) fail(err string) {
	panic(demangleErr{err: err, off: rst.off})
}

// advance advances the current string offset.
func (rst *rustState) advance(add int) {
	if len(rst.str) < add {
		panic("internal error")
	}
	rst.str = rst.str[add:]
	rst.off += add
}

// checkChar requires that the next character in the string be c,
// and advances past it.
func (rst *rustState) checkChar(c byte) {
	if len(rst.str) == 0 || rst.str[0] != c {
		rst.fail("expected " + string(c))
	}
	rst.advance(1)
}

// writeByte writes a byte to the buffer.
func (rst *rustState) writeByte(c byte) {
	if rst.skip {
		return
	}
	rst.last = c
	rst.buf.WriteByte(c)
}

// writeString writes a string to the buffer.
func (rst *rustState) writeString(s string) {
	if rst.skip {
		return
	}
	if len(s) > 0 {
		rst.last = s[len(s)-1]
		rst.buf.WriteString(s)
	}
}

// <symbol-name> = "_R" [<decimal-number>] <path> [<instantiating-crate>]
// <instantiating-crate> = <path>
//
// We've already skipped the "_R".
func (rst *rustState) symbolName() {
	if len(rst.str) < 1 {
		rst.fail("expected symbol-name")
	}

	if isDigit(rst.str[0]) {
		rst.fail("unsupported Rust encoding version")
	}

	rst.path(true)

	if len(rst.str) > 0 {
		rst.skip = true
		rst.path(false)
	}
}

// <path> = "C" <identifier>                    // crate root
//        | "M" <impl-path> <type>              // <T> (inherent impl)
//        | "X" <impl-path> <type> <path>       // <T as Trait> (trait impl)
//        | "Y" <type> <path>                   // <T as Trait> (trait definition)
//        | "N" <namespace> <path> <identifier> // ...::ident (nested path)
//        | "I" <path> {<generic-arg>} "E"      // ...<T, U> (generic args)
//        | <backref>
// <namespace> = "C"      // closure
//             | "S"      // shim
//             | <A-Z>    // other special namespaces
//             | <a-z>    // internal namespaces
//
// needsSeparator is true if we need to write out :: for a generic;
// it is passed as false if we are in the middle of a type.
func (rst *rustState) path(needsSeparator bool) {
	if len(rst.str) < 1 {
		rst.fail("expected path")
	}
	switch c := rst.str[0]; c {
	case 'C':
		rst.advance(1)
		_, ident := rst.identifier()
		rst.writeString(ident)
	case 'M', 'X':
		rst.advance(1)
		rst.implPath()
		rst.writeByte('<')
		rst.demangleType()
		if c == 'X' {
			rst.writeString(" as ")
			rst.path(false)
		}
		rst.writeByte('>')
	case 'Y':
		rst.advance(1)
		rst.writeByte('<')
		rst.demangleType()
		rst.writeString(" as ")
		rst.path(false)
		rst.writeByte('>')
	case 'N':
		rst.advance(1)

		if len(rst.str) < 1 {
			rst.fail("expected namespace")
		}
		ns := rst.str[0]
		switch {
		case ns >= 'a' && ns <= 'z':
		case ns >= 'A' && ns <= 'Z':
		default:
			rst.fail("invalid namespace character")
		}
		rst.advance(1)

		rst.path(needsSeparator)

		dis, ident := rst.identifier()

		if ns >= 'A' && ns <= 'Z' {
			rst.writeString("::{")
			switch ns {
			case 'C':
				rst.writeString("closure")
			case 'S':
				rst.writeString("shim")
			default:
				rst.writeByte(ns)
			}
			if len(ident) > 0 {
				rst.writeByte(':')
				rst.writeString(ident)
			}
			if !rst.skip {
				fmt.Fprintf(&rst.buf, "#%d}", dis)
				rst.last = '}'
			}
		} else {
			rst.writeString("::")
			rst.writeString(ident)
		}
	case 'I':
		rst.advance(1)
		rst.path(needsSeparator)
		if needsSeparator {
			rst.writeString("::")
		}
		rst.writeByte('<')
		first := true
		for len(rst.str) > 0 && rst.str[0] != 'E' {
			if first {
				first = false
			} else {
				rst.writeString(", ")
			}
			rst.genericArg()
		}
		rst.writeByte('>')
		rst.checkChar('E')
	case 'B':
		rst.backref(func() { rst.path(needsSeparator) })
	default:
		rst.fail("unrecognized letter in path")
	}
}

// <impl-path> = [<disambiguator>] <path>
func (rst *rustState) implPath() {
	// This path is not part of the demangled string.
	hold := rst.skip
	rst.skip = true
	defer func() {
		rst.skip = hold
	}()

	rst.disambiguator()
	rst.path(false)
}

// <identifier> = [<disambiguator>] <undisambiguated-identifier>
// Returns the disambiguator and the identifier.
func (rst *rustState) identifier() (int64, string) {
	dis := rst.disambiguator()
	ident := rst.undisambiguatedIdentifier()
	return dis, ident
}

// <disambiguator> = "s" <base-62-number>
// This is optional.
func (rst *rustState) disambiguator() int64 {
	if len(rst.str) == 0 || rst.str[0] != 's' {
		return 0
	}
	rst.advance(1)
	return rst.base62Number() + 1
}

// <undisambiguated-identifier> = ["u"] <decimal-number> ["_"] <bytes>
func (rst *rustState) undisambiguatedIdentifier() string {
	punycode := false
	if len(rst.str) > 0 && rst.str[0] == 'u' {
		rst.advance(1)
		punycode = true
	}

	val := rst.decimalNumber()

	if len(rst.str) > 0 && rst.str[0] == '_' {
		rst.advance(1)
	}

	if len(rst.str) < val {
		rst.fail("not enough characters for identifier")
	}
	id := rst.str[:val]
	rst.advance(val)

	for i := 0; i < len(id); i++ {
		c := id[i]
		switch {
		case c >= '0' && c <= '9':
		case c >= 'A' && c <= 'Z':
		case c >= 'a' && c <= 'z':
		case c == '_':
		default:
			rst.fail("invalid character in identifier")
		}
	}

	if punycode {
		id = rst.expandPunycode(id)
	}

	return id
}

// expandPunycode decodes the Rust version of punycode.
// This algorithm is taken from RFC 3492 section 6.2.
func (rst *rustState) expandPunycode(s string) string {
	const (
		base        = 36
		tmin        = 1
		tmax        = 26
		skew        = 38
		damp        = 700
		initialBias = 72
		initialN    = 128
	)

	idx := strings.LastIndex(s, "_")
	if idx < 0 {
		rst.fail("missing underscore in punycode string")
	}

	output := []rune(s[:idx])
	encoding := s[idx+1:]

	i := 0
	n := initialN
	bias := initialBias

	pos := 0
	for pos < len(encoding) {
		oldI := i
		w := 1
		for k := base; ; k += base {
			if pos == len(encoding) {
				rst.fail("unterminated punycode")
			}

			var digit byte
			d := encoding[pos]
			pos++
			switch {
			case '0' <= d && d <= '9':
				digit = d - '0' + 26
			case 'A' <= d && d <= 'Z':
				digit = d - 'A'
			case 'a' <= d && d <= 'z':
				digit = d - 'a'
			default:
				rst.fail("invalid punycode digit")
			}

			i += int(digit) * w
			if i < 0 {
				rst.fail("punycode number overflow")
			}

			var t int
			if k <= bias {
				t = tmin
			} else if k > bias+tmax {
				t = tmax
			} else {
				t = k - bias
			}

			if int(digit) < t {
				break
			}

			if w >= math.MaxInt32/base {
				rst.fail("punycode number overflow")
			}
			w *= base - t
		}

		delta := i - oldI
		numPoints := len(output) + 1
		firstTime := oldI == 0
		if firstTime {
			delta /= damp
		} else {
			delta /= 2
		}
		delta += delta / numPoints
		k := 0
		for delta > ((base-tmin)*tmax)/2 {
			delta /= base - tmin
			k += base
		}
		bias = k + ((base-tmin+1)*delta)/(delta+skew)

		n += i / (len(output) + 1)
		if n > utf8.MaxRune {
			rst.fail("punycode rune overflow")
		}
		i %= len(output) + 1
		output = append(output, 0)
		copy(output[i+1:], output[i:])
		output[i] = rune(n)
		i++
	}

	return string(output)
}

// <generic-arg> = <lifetime>
//               | <type>
//               | "K" <const> // forward-compat for const generics
// <lifetime> = "L" <base-62-number>
func (rst *rustState) genericArg() {
	if len(rst.str) < 1 {
		rst.fail("expected generic-arg")
	}
	if rst.str[0] == 'L' {
		rst.advance(1)
		rst.writeLifetime(rst.base62Number())
	} else if rst.str[0] == 'K' {
		rst.advance(1)
		rst.demangleConst()
	} else {
		rst.demangleType()
	}
}

// <binder> = "G" <base-62-number>
// This is optional.
func (rst *rustState) binder() {
	if len(rst.str) < 1 || rst.str[0] != 'G' {
		return
	}
	rst.advance(1)

	binderLifetimes := rst.base62Number() + 1

	// Every bound lifetime should be referenced later.
	if binderLifetimes >= int64(len(rst.str))-rst.lifetimes {
		rst.fail("binder lifetimes overflow")
	}

	rst.writeString("for<")
	for i := int64(0); i < binderLifetimes; i++ {
		if i > 0 {
			rst.writeString(", ")
		}
		rst.lifetimes++
		rst.writeLifetime(1)
	}
	rst.writeString("> ")
}

// <type> = <basic-type>
//        | <path>                      // named type
//        | "A" <type> <const>          // [T; N]
//        | "S" <type>                  // [T]
//        | "T" {<type>} "E"            // (T1, T2, T3, ...)
//        | "R" [<lifetime>] <type>     // &T
//        | "Q" [<lifetime>] <type>     // &mut T
//        | "P" <type>                  // *const T
//        | "O" <type>                  // *mut T
//        | "F" <fn-sig>                // fn(...) -> ...
//        | "D" <dyn-bounds> <lifetime> // dyn Trait<Assoc = X> + Send + 'a
//        | <backref>
func (rst *rustState) demangleType() {
	if len(rst.str) < 1 {
		rst.fail("expected type")
	}
	c := rst.str[0]
	if c >= 'a' && c <= 'z' {
		rst.basicType()
		return
	}
	switch c {
	case 'C', 'M', 'X', 'Y', 'N', 'I':
		rst.path(false)
	case 'A', 'S':
		rst.advance(1)
		rst.writeByte('[')
		rst.demangleType()
		if c == 'A' {
			rst.writeString("; ")
			rst.demangleConst()
		}
		rst.writeByte(']')
	case 'T':
		rst.advance(1)
		rst.writeByte('(')
		c := 0
		for len(rst.str) > 0 && rst.str[0] != 'E' {
			if c > 0 {
				rst.writeString(", ")
			}
			c++
			rst.demangleType()
		}
		if c == 1 {
			rst.writeByte(',')
		}
		rst.writeByte(')')
		rst.checkChar('E')
	case 'R', 'Q':
		rst.advance(1)
		rst.writeByte('&')
		if len(rst.str) > 0 && rst.str[0] == 'L' {
			rst.advance(1)
			if lifetime := rst.base62Number(); lifetime > 0 {
				rst.writeLifetime(lifetime)
				rst.writeByte(' ')
			}
		}
		if c == 'Q' {
			rst.writeString("mut ")
		}
		rst.demangleType()
	case 'P':
		rst.advance(1)
		rst.writeString("*const ")
		rst.demangleType()
	case 'O':
		rst.advance(1)
		rst.writeString("*mut ")
		rst.demangleType()
	case 'F':
		rst.advance(1)
		hold := rst.lifetimes
		rst.fnSig()
		rst.lifetimes = hold
	case 'D':
		rst.advance(1)
		hold := rst.lifetimes
		rst.dynBounds()
		rst.lifetimes = hold
		if len(rst.str) == 0 || rst.str[0] != 'L' {
			rst.fail("expected L")
		}
		rst.advance(1)
		if lifetime := rst.base62Number(); lifetime > 0 {
			if rst.last != ' ' {
				rst.writeByte(' ')
			}
			rst.writeString("+ ")
			rst.writeLifetime(lifetime)
		}
	case 'B':
		rst.backref(rst.demangleType)
	default:
		rst.fail("unrecognized character in type")
	}
}

var rustBasicTypes = map[byte]string{
	'a': "i8",
	'b': "bool",
	'c': "char",
	'd': "f64",
	'e': "str",
	'f': "f32",
	'h': "u8",
	'i': "isize",
	'j': "usize",
	'l': "i32",
	'm': "u32",
	'n': "i128",
	'o': "u128",
	'p': "_",
	's': "i16",
	't': "u16",
	'u': "()",
	'v': "...",
	'x': "i64",
	'y': "u64",
	'z': "!",
}

// <basic-type>
func (rst *rustState) basicType() {
	if len(rst.str) < 1 {
		rst.fail("expected basic type")
	}
	str, ok := rustBasicTypes[rst.str[0]]
	if !ok {
		rst.fail("unrecognized basic type character")
	}
	rst.advance(1)
	rst.writeString(str)
}

// <fn-sig> = [<binder>] ["U"] ["K" <abi>] {<type>} "E" <type>
// <abi> = "C"
//       | <undisambiguated-identifier>
func (rst *rustState) fnSig() {
	rst.binder()
	if len(rst.str) > 0 && rst.str[0] == 'U' {
		rst.advance(1)
		rst.writeString("unsafe ")
	}
	if len(rst.str) > 0 && rst.str[0] == 'K' {
		rst.advance(1)
		if len(rst.str) > 0 && rst.str[0] == 'C' {
			rst.advance(1)
			rst.writeString(`extern "C" `)
		} else {
			rst.writeString(`extern "`)
			id := rst.undisambiguatedIdentifier()
			id = strings.ReplaceAll(id, "_", "-")
			rst.writeString(id)
			rst.writeString(`" `)
		}
	}
	rst.writeString("fn(")
	first := true
	for len(rst.str) > 0 && rst.str[0] != 'E' {
		if first {
			first = false
		} else {
			rst.writeString(", ")
		}
		rst.demangleType()
	}
	rst.checkChar('E')
	rst.writeByte(')')
	if len(rst.str) > 0 && rst.str[0] == 'u' {
		rst.advance(1)
	} else {
		rst.writeString(" -> ")
		rst.demangleType()
	}
}

// <dyn-bounds> = [<binder>] {<dyn-trait>} "E"
func (rst *rustState) dynBounds() {
	rst.writeString("dyn ")
	rst.binder()
	first := true
	for len(rst.str) > 0 && rst.str[0] != 'E' {
		if first {
			first = false
		} else {
			rst.writeString(" + ")
		}
		rst.dynTrait()
	}
	rst.checkChar('E')
}

// <dyn-trait> = <path> {<dyn-trait-assoc-binding>}
// <dyn-trait-assoc-binding> = "p" <undisambiguated-identifier> <type>
func (rst *rustState) dynTrait() {
	started := rst.pathStartGenerics()
	for len(rst.str) > 0 && rst.str[0] == 'p' {
		rst.advance(1)
		if started {
			rst.writeString(", ")
		} else {
			rst.writeByte('<')
			started = true
		}
		rst.writeString(rst.undisambiguatedIdentifier())
		rst.writeString(" = ")
		rst.demangleType()
	}
	if started {
		rst.writeByte('>')
	}
}

// pathStartGenerics is like path but if it sees an I to start generic
// arguments it won't close them. It reports whether it started generics.
func (rst *rustState) pathStartGenerics() bool {
	if len(rst.str) < 1 {
		rst.fail("expected path")
	}
	switch rst.str[0] {
	case 'I':
		rst.advance(1)
		rst.path(false)
		rst.writeByte('<')
		first := true
		for len(rst.str) > 0 && rst.str[0] != 'E' {
			if first {
				first = false
			} else {
				rst.writeString(", ")
			}
			rst.genericArg()
		}
		rst.checkChar('E')
		return true
	case 'B':
		var started bool
		rst.backref(func() { started = rst.pathStartGenerics() })
		return started
	default:
		rst.path(false)
		return false
	}
}

// writeLifetime writes out a lifetime binding.
func (rst *rustState) writeLifetime(lifetime int64) {
	rst.writeByte('\'')
	if lifetime == 0 {
		rst.writeByte('_')
		return
	}
	depth := rst.lifetimes - lifetime
	if depth < 0 {
		rst.fail("invalid lifetime")
	} else if depth < 26 {
		rst.writeByte('a' + byte(depth))
	} else {
		rst.writeByte('z')
		if !rst.skip {
			fmt.Fprintf(&rst.buf, "%d", depth-26+1)
			rst.last = '0'
		}
	}
}

// <const> = <type> <const-data>
//         | "p" // placeholder, shown as _
//         | <backref>
// <const-data> = ["n"] {<hex-digit>} "_"
func (rst *rustState) demangleConst() {
	if len(rst.str) < 1 {
		rst.fail("expected constant")
	}

	if rst.str[0] == 'B' {
		rst.backref(rst.demangleConst)
		return
	}

	if rst.str[0] == 'p' {
		rst.advance(1)
		rst.writeByte('_')
		return
	}

	typ := rst.str[0]

	const (
		invalid = iota
		signedInt
		unsignedInt
		boolean
		character
	)

	var kind int
	switch typ {
	case 'a', 's', 'l', 'x', 'n', 'i':
		kind = signedInt
	case 'h', 't', 'm', 'y', 'o', 'j':
		kind = unsignedInt
	case 'b':
		kind = boolean
	case 'c':
		kind = character
	default:
		rst.fail("unrecognized constant type")
	}

	rst.advance(1)

	if kind == signedInt && len(rst.str) > 0 && rst.str[0] == 'n' {
		rst.advance(1)
		rst.writeByte('-')
	}

	start := rst.str
	digits := 0
	val := uint64(0)
digitLoop:
	for len(rst.str) > 0 {
		c := rst.str[0]
		var digit uint64
		switch {
		case c >= '0' && c <= '9':
			digit = uint64(c - '0')
		case c >= 'a' && c <= 'f':
			digit = uint64(c - 'a' + 10)
		case c == '_':
			rst.advance(1)
			break digitLoop
		default:
			rst.fail("expected hex digit or _")
		}
		rst.advance(1)
		if val == 0 && digit == 0 && (len(rst.str) == 0 || rst.str[0] != '_') {
			rst.fail("invalid leading 0 in constant")
		}
		val *= 16
		val += digit
		digits++
	}

	if digits == 0 {
		rst.fail("expected constant")
	}

	switch kind {
	case signedInt, unsignedInt:
		if digits > 16 {
			// Value too big, just write out the string.
			rst.writeString("0x")
			rst.writeString(start[:digits])
		} else {
			if !rst.skip {
				fmt.Fprintf(&rst.buf, "%d", val)
				rst.last = '0'
			}
		}
	case boolean:
		if digits > 1 {
			rst.fail("boolean value too large")
		} else if val == 0 {
			rst.writeString("false")
		} else if val == 1 {
			rst.writeString("true")
		} else {
			rst.fail("invalid boolean value")
		}
	case character:
		if digits > 6 {
			rst.fail("character value too large")
		}
		rst.writeByte('\'')
		if val == '\t' {
			rst.writeString(`\t`)
		} else if val == '\r' {
			rst.writeString(`\r`)
		} else if val == '\n' {
			rst.writeString(`\n`)
		} else if val == '\\' {
			rst.writeString(`\\`)
		} else if val == '\'' {
			rst.writeString(`\'`)
		} else if val >= ' ' && val <= '~' {
			// printable ASCII character
			rst.writeByte(byte(val))
		} else {
			if !rst.skip {
				fmt.Fprintf(&rst.buf, `\u{%x}`, val)
				rst.last = '}'
			}
		}
		rst.writeByte('\'')
	default:
		panic("internal error")
	}
}

// <base-62-number> = {<0-9a-zA-Z>} "_"
func (rst *rustState) base62Number() int64 {
	if len(rst.str) > 0 && rst.str[0] == '_' {
		rst.advance(1)
		return 0
	}
	val := int64(0)
	for len(rst.str) > 0 {
		c := rst.str[0]
		rst.advance(1)
		if c == '_' {
			return val + 1
		}
		val *= 62
		if c >= '0' && c <= '9' {
			val += int64(c - '0')
		} else if c >= 'a' && c <= 'z' {
			val += int64(c - 'a' + 10)
		} else if c >= 'A' && c <= 'Z' {
			val += int64(c - 'A' + 36)
		} else {
			rst.fail("invalid digit in base 62 number")
		}
	}
	rst.fail("expected _ after base 62 number")
	return 0
}

// <backref> = "B" <base-62-number>
func (rst *rustState) backref(demangle func()) {
	backoff := rst.off

	rst.checkChar('B')
	idx64 := rst.base62Number()

	if rst.skip {
		return
	}

	idx := int(idx64)
	if int64(idx) != idx64 {
		rst.fail("backref index overflow")
	}
	if idx < 0 || idx >= backoff {
		rst.fail("invalid backref index")
	}

	holdStr := rst.str
	holdOff := rst.off
	rst.str = rst.orig[idx:backoff]
	rst.off = idx
	defer func() {
		rst.str = holdStr
		rst.off = holdOff
	}()

	demangle()
}

func (rst *rustState) decimalNumber() int {
	if len(rst.str) == 0 {
		rst.fail("expected number")
	}

	val := 0
	for len(rst.str) > 0 && isDigit(rst.str[0]) {
		add := int(rst.str[0] - '0')
		if val >= math.MaxInt32/10-add {
			rst.fail("decimal number overflow")
		}
		val *= 10
		val += add
		rst.advance(1)
	}
	return val
}

// oldRustToString demangles a Rust symbol using the old demangling.
// The second result reports whether this is a valid Rust mangled name.
func oldRustToString(name string, options []Option) (string, bool) {
	// We know that the string starts with _ZN.
	name = name[3:]

	hexDigit := func(c byte) (byte, bool) {
		switch {
		case c >= '0' && c <= '9':
			return c - '0', true
		case c >= 'a' && c <= 'f':
			return c - 'a' + 10, true
		default:
			return 0, false
		}
	}

	// We know that the strings end with "17h" followed by 16 characters
	// followed by "E". We check that the 16 characters are all hex digits.
	// Also the hex digits must contain at least 5 distinct digits.
	seen := uint16(0)
	for i := len(name) - 17; i < len(name) - 1; i++ {
		digit, ok := hexDigit(name[i])
		if !ok {
			return "", false
		}
		seen |= 1 << digit
	}
	if bits.OnesCount16(seen) < 5 {
		return "", false
	}
	name = name[:len(name)-20]

	// The name is a sequence of length-preceded identifiers.
	var sb strings.Builder
	for len(name) > 0 {
		if !isDigit(name[0]) {
			return "", false
		}

		val := 0
		for len(name) > 0 && isDigit(name[0]) {
			add := int(name[0] - '0')
			if val >= math.MaxInt32/10-add {
				return "", false
			}
			val *= 10
			val += add
			name = name[1:]
		}

		// An optional trailing underscore can separate the
		// length from the identifier.
		if len(name) > 0 && name[0] == '_' {
			name = name[1:]
			val--
		}

		if len(name) < val {
			return "", false
		}

		id := name[:val]
		name = name[val:]

		if sb.Len() > 0 {
			sb.WriteString("::")
		}

		// Ignore leading underscores preceding escape sequences.
		if strings.HasPrefix(id, "_$") {
			id = id[1:]
		}

		// The identifier can have escape sequences.
	escape:
		for len(id) > 0 {
			switch c := id[0]; c {
			case '$':
				codes := map[string]byte {
					"SP": '@',
					"BP": '*',
					"RF": '&',
					"LT": '<',
					"GT": '>',
					"LP": '(',
					"RP": ')',
				}

				valid := true
				if len(id) > 2 && id[1] == 'C' && id[2] == '$' {
					sb.WriteByte(',')
					id = id[3:]
				} else if len(id) > 4 && id[1] == 'u' && id[4] == '$' {
					dig1, ok1 := hexDigit(id[2])
					dig2, ok2 := hexDigit(id[3])
					val := (dig1 << 4) | dig2
					if !ok1 || !ok2 || dig1 > 7 || val < ' ' {
						valid = false
					} else {
						sb.WriteByte(val)
						id = id[5:]
					}
				} else if len(id) > 3 && id[3] == '$' {
					if code, ok := codes[id[1:3]]; !ok {
						valid = false
					} else {
						sb.WriteByte(code)
						id = id[4:]
					}
				} else {
					valid = false
				}
				if !valid {
					sb.WriteString(id)
					break escape
				}
			case '.':
				if strings.HasPrefix(id, "..") {
					sb.WriteString("::")
					id = id[2:]
				} else {
					sb.WriteByte(c)
					id = id[1:]
				}
			default:
				sb.WriteByte(c)
				id = id[1:]
			}
		}
	}

	return sb.String(), true
}
