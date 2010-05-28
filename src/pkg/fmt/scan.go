// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"bytes"
	"io"
	"os"
	"reflect"
	"strconv"
	"unicode"
	"utf8"
)

// readRuner is the interface to something that can read runes.  If
// the object provided to Scan does not satisfy this interface, the
// object will be wrapped by a readRune object.
type readRuner interface {
	ReadRune() (rune int, size int, err os.Error)
}

// ScanState represents the scanner state passed to custom scanners.
// Scanners may do rune-at-a-time scanning or ask the ScanState
// to discover the next space-delimited token.
type ScanState interface {
	// GetRune reads the next rune (Unicode code point) from the input.
	GetRune() (rune int, err os.Error)
	// UngetRune causes the next call to Get to return the rune.
	UngetRune(rune int)
	// Token returns the next space-delimited token from the input.
	Token() (token string, err os.Error)
}

// Scanner is implemented by any value that has a Scan method, which scans
// the input for the representation of a value and stores the result in the
// receiver, which must be a pointer to be useful.  The Scan method is called
// for any argument to Scan or Scanln that implements it.
type Scanner interface {
	Scan(ScanState) os.Error
}

// ss is the internal implementation of ScanState.
type ss struct {
	rr        readRuner    // where to read input
	buf       bytes.Buffer // token accumulator
	nlIsSpace bool         // whether newline counts as white space
	peekRune  int          // one-rune lookahead
	err       os.Error
}

func (s *ss) GetRune() (rune int, err os.Error) {
	if s.peekRune >= 0 {
		rune = s.peekRune
		s.peekRune = -1
		return
	}
	rune, _, err = s.rr.ReadRune()
	return
}

func (s *ss) UngetRune(rune int) {
	s.peekRune = rune
}

func (s *ss) Token() (tok string, err os.Error) {
	tok = s.token()
	err = s.err
	return
}

// readRune is a structure to enable reading UTF-8 encoded code points
// from an io.Reader.  It is used if the Reader given to the scanner does
// not already implement ReadRuner.
// TODO: readByteRune for things that can read bytes.
type readRune struct {
	reader io.Reader
	buf    [utf8.UTFMax]byte
}

// ReadRune returns the next UTF-8 encoded code point from the
// io.Reader inside r.
func (r readRune) ReadRune() (rune int, size int, err os.Error) {
	_, err = r.reader.Read(r.buf[0:1])
	if err != nil {
		return 0, 0, err
	}
	if r.buf[0] < utf8.RuneSelf { // fast check for common ASCII case
		rune = int(r.buf[0])
		return
	}
	for size := 1; size < utf8.UTFMax; size++ {
		_, err = r.reader.Read(r.buf[size : size+1])
		if err != nil {
			break
		}
		if !utf8.FullRune(r.buf[0:]) {
			continue
		}
		if c, w := utf8.DecodeRune(r.buf[0:size]); w == size {
			rune = c
			return
		}
	}
	return utf8.RuneError, 1, err
}


// A leaky bucket of reusable ss structures.
var ssFree = make(chan *ss, 100)

// Allocate a new ss struct.  Probably can grab the previous one from ssFree.
func newScanState(r io.Reader, nlIsSpace bool) *ss {
	s, ok := <-ssFree
	if !ok {
		s = new(ss)
	}
	if rr, ok := r.(readRuner); ok {
		s.rr = rr
	} else {
		s.rr = readRune{reader: r}
	}
	s.nlIsSpace = nlIsSpace
	s.peekRune = -1
	s.err = nil
	return s
}

// Save used ss structs in ssFree; avoid an allocation per invocation.
func (s *ss) free() {
	// Don't hold on to ss structs with large buffers.
	if cap(s.buf.Bytes()) > 1024 {
		return
	}
	s.buf.Reset()
	s.rr = nil
	_ = ssFree <- s
}

// token returns the next space-delimited string from the input.
// For Scanln, it stops at newlines.  For Scan, newlines are treated as
// spaces.
func (s *ss) token() string {
	s.buf.Reset()
	// skip white space and maybe newline
	for {
		rune, err := s.GetRune()
		if err != nil {
			s.err = err
			return ""
		}
		if rune == '\n' {
			if s.nlIsSpace {
				continue
			}
			s.err = os.ErrorString("unexpected newline")
			return ""
		}
		if !unicode.IsSpace(rune) {
			s.buf.WriteRune(rune)
			break
		}
	}
	// read until white space or newline
	for {
		rune, err := s.GetRune()
		if err != nil {
			if err == os.EOF {
				break
			}
			s.err = err
			return ""
		}
		if unicode.IsSpace(rune) {
			s.UngetRune(rune)
			break
		}
		s.buf.WriteRune(rune)
	}
	return s.buf.String()
}

// Scan parses text read from standard input, storing successive
// space-separated values into successive arguments.  Newlines count as
// space.  Each argument must be a pointer to a basic type or an
// implementation of the Scanner interface.  It returns the number of items
// successfully parsed.  If that is less than the number of arguments, err
// will report why.
func Scan(a ...interface{}) (n int, err os.Error) {
	return Fscan(os.Stdin, a)
}

// Fscanln parses text read from standard input, storing successive
// space-separated values into successive arguments.  Scanning stops at a
// newline and after the final item there must be a newline or EOF.  Each
// argument must be a pointer to a basic type or an implementation of the
// Scanner interface.  It returns the number of items successfully parsed.
// If that is less than the number of arguments, err will report why.
func Scanln(a ...interface{}) (n int, err os.Error) {
	return Fscanln(os.Stdin, a)
}

// Fscan parses text read from r, storing successive space-separated values
// into successive arguments.  Newlines count as space.  Each argument must
// be a pointer to a basic type or an implementation of the Scanner
// interface.  It returns the number of items successfully parsed.  If that
// is less than the number of arguments, err will report why.
func Fscan(r io.Reader, a ...interface{}) (n int, err os.Error) {
	s := newScanState(r, true)
	n = s.doScan(a)
	err = s.err
	s.free()
	return
}

// Fscanln parses text read from r, storing successive space-separated values
// into successive arguments.  Scanning stops at a newline and after the
// final item there must be a newline or EOF.  Each argument must be a
// pointer to a basic type or an implementation of the Scanner interface.  It
// returns the number of items successfully parsed.  If that is less than the
// number of arguments, err will report why.
func Fscanln(r io.Reader, a ...interface{}) (n int, err os.Error) {
	s := newScanState(r, false)
	n = s.doScan(a)
	err = s.err
	s.free()
	return
}

var intBits = uint(reflect.Typeof(int(0)).Size() * 8)
var uintptrBits = uint(reflect.Typeof(int(0)).Size() * 8)
var complexError = os.ErrorString("syntax error scanning complex number")

// scanBool converts the token to a boolean value.
func (s *ss) scanBool(tok string) bool {
	if s.err != nil {
		return false
	}
	var b bool
	b, s.err = strconv.Atob(tok)
	return b
}

// complexParts returns the strings representing the real and imaginary parts of the string.
func (s *ss) complexParts(str string) (real, imag string) {
	if len(str) > 2 && str[0] == '(' && str[len(str)-1] == ')' {
		str = str[1 : len(str)-1]
	}
	real, str = floatPart(str)
	// Must now have a sign.
	if len(str) == 0 || (str[0] != '+' && str[0] != '-') {
		s.err = complexError
		return "", ""
	}
	imag, str = floatPart(str)
	if str != "i" {
		s.err = complexError
		return "", ""
	}
	return real, imag
}

// floatPart returns strings holding the floating point value in the string, followed
// by the remainder of the string.  That is, it splits str into (number,rest-of-string).
func floatPart(str string) (first, last string) {
	i := 0
	// leading sign?
	if len(str) > 0 && (str[0] == '+' || str[0] == '-') {
		i++
	}
	// digits?
	for len(str) > 0 && '0' <= str[i] && str[i] <= '9' {
		i++
	}
	// period?
	if str[i] == '.' {
		i++
	}
	// fraction?
	for len(str) > 0 && '0' <= str[i] && str[i] <= '9' {
		i++
	}
	// exponent?
	if len(str) > 0 && (str[i] == 'e' || str[i] == 'E') {
		i++
		// leading sign?
		if str[0] == '+' || str[0] == '-' {
			i++
		}
		// digits?
		for len(str) > 0 && '0' <= str[i] && str[i] <= '9' {
			i++
		}
	}
	return str[0:i], str[i:]
}

// scanFloat converts the string to a float value.
func (s *ss) scanFloat(str string) float64 {
	var f float
	f, s.err = strconv.Atof(str)
	return float64(f)
}

// scanFloat32 converts the string to a float32 value.
func (s *ss) scanFloat32(str string) float64 {
	var f float32
	f, s.err = strconv.Atof32(str)
	return float64(f)
}

// scanFloat64 converts the string to a float64 value.
func (s *ss) scanFloat64(str string) float64 {
	var f float64
	f, s.err = strconv.Atof64(str)
	return f
}

// scanComplex converts the token to a complex128 value.
// The atof argument is a type-specific reader for the underlying type.
// If we're reading complex64, atof will parse float32s and convert them
// to float64's to avoid reproducing this code for each complex type.
func (s *ss) scanComplex(tok string, atof func(*ss, string) float64) complex128 {
	if s.err != nil {
		return 0
	}
	sreal, simag := s.complexParts(tok)
	if s.err != nil {
		return 0
	}
	var real, imag float64
	real = atof(s, sreal)
	if s.err != nil {
		return 0
	}
	imag = atof(s, simag)
	if s.err != nil {
		return 0
	}
	return cmplx(real, imag)
}

// scanInt converts the token to an int64, but checks that it fits into the
// specified number of bits.
func (s *ss) scanInt(tok string, bitSize uint) int64 {
	if s.err != nil {
		return 0
	}
	var i int64
	i, s.err = strconv.Atoi64(tok)
	x := (i << (64 - bitSize)) >> (64 - bitSize)
	if i != x {
		s.err = os.ErrorString("integer overflow on token " + tok)
	}
	return i
}

// scanUint converts the token to a uint64, but checks that it fits into the
// specified number of bits.
func (s *ss) scanUint(tok string, bitSize uint) uint64 {
	if s.err != nil {
		return 0
	}
	var i uint64
	i, s.err = strconv.Atoui64(tok)
	x := (i << (64 - bitSize)) >> (64 - bitSize)
	if i != x {
		s.err = os.ErrorString("unsigned integer overflow on token " + tok)
	}
	return i
}

// doScan does the real work.  At the moment, it handles only pointers to basic types.
func (s *ss) doScan(a []interface{}) int {
	for n, param := range a {
		// If the parameter has its own Scan method, use that.
		if v, ok := param.(Scanner); ok {
			s.err = v.Scan(s)
			if s.err != nil {
				return n
			}
			continue
		}
		tok := s.token()
		switch v := param.(type) {
		case *bool:
			*v = s.scanBool(tok)
		case *complex:
			*v = complex(s.scanComplex(tok, (*ss).scanFloat))
		case *complex64:
			*v = complex64(s.scanComplex(tok, (*ss).scanFloat32))
		case *complex128:
			*v = s.scanComplex(tok, (*ss).scanFloat64)
		case *int:
			*v = int(s.scanInt(tok, intBits))
		case *int8:
			*v = int8(s.scanInt(tok, 8))
		case *int16:
			*v = int16(s.scanInt(tok, 16))
		case *int32:
			*v = int32(s.scanInt(tok, 32))
		case *int64:
			*v = s.scanInt(tok, 64)
		case *uint:
			*v = uint(s.scanUint(tok, intBits))
		case *uint8:
			*v = uint8(s.scanUint(tok, 8))
		case *uint16:
			*v = uint16(s.scanUint(tok, 16))
		case *uint32:
			*v = uint32(s.scanUint(tok, 32))
		case *uint64:
			*v = s.scanUint(tok, 64)
		case *uintptr:
			*v = uintptr(s.scanUint(tok, uintptrBits))
		case *float:
			if s.err == nil {
				*v, s.err = strconv.Atof(tok)
			} else {
				*v = 0
			}
		case *float32:
			if s.err == nil {
				*v, s.err = strconv.Atof32(tok)
			} else {
				*v = 0
			}
		case *float64:
			if s.err == nil {
				*v, s.err = strconv.Atof64(tok)
			} else {
				*v = 0
			}
		case *string:
			*v = tok
		default:
			t := reflect.Typeof(v)
			str := t.String()
			if _, ok := t.(*reflect.PtrType); !ok {
				s.err = os.ErrorString("Scan: type not a pointer: " + str)
			} else {
				s.err = os.ErrorString("Scan: can't handle type: " + str)
			}
		}
		if s.err != nil {
			return n
		}
	}
	// Check for newline if required.
	if !s.nlIsSpace {
		for {
			rune, err := s.GetRune()
			if err != nil {
				if err == os.EOF {
					break
				}
				s.err = err
				break
			}
			if rune == '\n' {
				break
			}
			if !unicode.IsSpace(rune) {
				s.err = os.ErrorString("Scan: expected newline")
				break
			}
		}
	}
	return len(a)
}
