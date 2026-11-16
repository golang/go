// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"encoding"
	stdjson "encoding/json"
	"encoding/json/jsontext"
	"errors"
	"io"
	"reflect"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

const indexedFieldPaths = true

var stringContentReplacer = strings.NewReplacer(
	"object name", "object key",
	"at start of value", "looking for beginning of value",
	"at start of string", "looking for beginning of object key string",
	"after object value", "after object key:value pair",
	"in number", "in numeric literal",
)

func stringContentError(item string) error {
	_, err := jsontext.AppendUnquote(nil, []byte(item))
	if err == nil {
		return nil
	}
	msg := err.Error()
	if syn, ok := err.(*jsontext.SyntacticError); ok && syn.Err != nil {
		msg = syn.Err.Error()
	}
	if msg == io.ErrUnexpectedEOF.Error() {
		return errors.New("unexpected end of JSON input")
	}
	if before, _, found := strings.Cut(msg, " (expecting"); found && !strings.Contains(msg, " in literal") {
		msg = before
	}
	return errors.New(stringContentReplacer.Replace(msg))
}

func (s *Scanner) rejectQuotedBare(typ reflect.Type) {
	s.rejectKind(typ)
}

func (s *Scanner) rejectQuotedContent(res quotedResult, item string, typ reflect.Type) {
	switch typ.Kind() {
	case reflect.String:
		if contentErr := stringContentError(item); contentErr != nil {
			ute := s.typeError("string", typ)
			ute.Err = contentErr
			s.save(ute)
		} else {
			s.rejectKind(typ)
		}
	case reflect.Bool:
		ute := s.typeError("string "+s.literal(), typ)
		ute.Err = strconv.ErrSyntax
		s.save(ute)
	default:
		s.rejectNumber(item, typ)
	}
}

func quotedIntFallback(item string, bits int) (int64, bool) {
	v, err := strconv.ParseInt(item, 10, bits)
	if err != nil {
		return 0, false
	}
	return v, true
}

func quotedUintFallback(item string, bits int) (uint64, bool) {
	v, err := strconv.ParseUint(item, 10, bits)
	if err != nil {
		return 0, false
	}
	return v, true
}

func quotedFloatFallback(item string, bits int) (float64, bool) {
	v, err := strconv.ParseFloat(item, bits)
	if err != nil {
		return 0, false
	}
	return v, true
}

func unquoteStringContent(item string) (string, bool) {
	v, err := jsontext.AppendUnquote(nil, item)
	if err != nil {
		return "", false
	}
	return unsafeString(v), true
}

var quotedNullLiteral = []byte(`"null"`)

// rejectMethodUnsupported mirrors wrapErrUnsupported: only streaming
// methods may decline a value with errors.ErrUnsupported; a classic
// method returning it reports a misuse error instead.
func rejectMethodUnsupported(err error, what string) error {
	if errors.Is(err, errors.ErrUnsupported) {
		return errors.New(what + " may not return errors.ErrUnsupported")
	}
	return err
}

func QuotedJSON[T stdjson.Unmarshaler](s *Scanner, dst T) {
	if s.scalar() == 0 {
		return
	}
	// Cap the slice so an appending method cannot overwrite the rest of
	// the document.
	tok := s.data[s.start:s.end:s.end]
	s.saveMethodError(rejectMethodUnsupported(dst.UnmarshalJSON(tok), "UnmarshalJSON method"))
}

func QuotedJSONPtr[E any, T interface {
	*E
	stdjson.Unmarshaler
}](s *Scanner, dst *T) {
	if s.scalar() == 0 {
		return
	}
	tok := s.data[s.start:s.end:s.end]
	if s.kind == 'n' {
		*dst = nil
		return
	}
	if *dst == nil {
		*dst = new(E)
	}
	err := rejectMethodUnsupported((*dst).UnmarshalJSON(tok), "UnmarshalJSON method")
	s.saveMethodError(err)
	if err == nil && bytes.Equal(tok, quotedNullLiteral) {
		*dst = nil
	}
}

// errNonStringValue mirrors the message encoding/json reports through
// UnmarshalTypeError.Err when a `,string` field's method sees a non-string
// value.
var errNonStringValue = errors.New("JSON value must be string type")

// rejectNonString saves the type error a `,string` field reports for a
// non-string value. The token may not reach a further unmarshal method:
// a fresh Unmarshal call would dispatch the type's UnmarshalJSONFrom a
// second time even though it already declined the value.
func (s *Scanner) rejectNonString(tok []byte, typ reflect.Type) {
	var value string
	switch tok[0] {
	case '{':
		value = "object"
	case '[':
		value = "array"
	case 't', 'f':
		value = "bool"
	default:
		value = "number"
	}
	off := int64(len(tok))
	if value == "object" || value == "array" {
		off = 1
	}
	s.saveFallback(&stdjson.UnmarshalTypeError{
		Value: value, Type: typ, Offset: off, Err: errNonStringValue})
}

func QuotedText[T encoding.TextUnmarshaler](s *Scanner, dst T) {
	if s.scalar() == 0 {
		return
	}
	tok := s.data[s.start:s.end:s.end]
	if s.kind == 'n' {
		return
	}
	if len(tok) > 0 && tok[0] == '"' {
		if unq, ok := permissiveUnquote(tok); ok {
			s.saveMethodError(rejectMethodUnsupported(dst.UnmarshalText(unq), "UnmarshalText method"))
			return
		}
	}
	typ := reflect.TypeFor[T]()
	if typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	s.rejectNonString(tok, typ)
}

func QuotedTextPtr[E any, T interface {
	*E
	encoding.TextUnmarshaler
}](s *Scanner, dst *T) {
	if s.scalar() == 0 {
		return
	}
	tok := s.data[s.start:s.end:s.end]
	if s.kind == 'n' {
		*dst = nil
		return
	}
	if len(tok) > 0 && tok[0] == '"' {
		if *dst == nil {
			*dst = new(E)
		}
		if unq, ok := permissiveUnquote(tok); ok {
			err := rejectMethodUnsupported((*dst).UnmarshalText(unq), "UnmarshalText method")
			s.saveMethodError(err)
			if err == nil && bytes.Equal(tok, quotedNullLiteral) {
				*dst = nil
			}
			return
		}
	}
	if *dst == nil {
		*dst = new(E)
	}
	s.rejectNonString(tok, reflect.TypeFor[E]())
}

// permissiveUnquote mirrors encoding/json unquoteBytes for the `,string` text
// methods: malformed UTF-8 bytes and unpaired surrogates decode as U+FFFD
// instead of rejecting the string token, so the method sees the same bytes
// the legacy profile would pass it. jsontext.AppendUnquote is strict about
// both and must not gate method dispatch here.
func permissiveUnquote(tok []byte) ([]byte, bool) {
	if len(tok) < 2 || tok[0] != '"' || tok[len(tok)-1] != '"' {
		return nil, false
	}
	s := tok[1 : len(tok)-1]

	r := 0
	for r < len(s) {
		c := s[r]
		if c == '\\' || c == '"' || c < ' ' {
			break
		}
		rr, size := utf8.DecodeRune(s[r:])
		if rr == utf8.RuneError && size == 1 {
			break
		}
		r += size
	}
	if r == len(s) {
		// Cap the fast path so an UnmarshalText that appends to its input
		// allocates instead of overwriting the token's closing quote, which
		// QuotedTextPtr still compares against quotedNullLiteral.
		return s[:len(s):len(s)], true
	}

	b := make([]byte, len(s)+2*utf8.UTFMax)
	w := copy(b, s[:r])
	for r < len(s) {
		if w >= len(b)-2*utf8.UTFMax {
			nb := make([]byte, (len(b)+utf8.UTFMax)*2)
			copy(nb, b[:w])
			b = nb
		}
		switch c := s[r]; {
		case c == '\\':
			r++
			if r >= len(s) {
				return nil, false
			}
			switch s[r] {
			default:
				return nil, false
			case '"', '\\', '/', '\'':
				b[w] = s[r]
				r++
				w++
			case 'b':
				b[w] = '\b'
				r++
				w++
			case 'f':
				b[w] = '\f'
				r++
				w++
			case 'n':
				b[w] = '\n'
				r++
				w++
			case 'r':
				b[w] = '\r'
				r++
				w++
			case 't':
				b[w] = '\t'
				r++
				w++
			case 'u':
				r--
				rr := getu4(s[r:])
				if rr < 0 {
					return nil, false
				}
				r += 6
				if utf16.IsSurrogate(rr) {
					rr1 := getu4(s[r:])
					if dec := utf16.DecodeRune(rr, rr1); dec != unicode.ReplacementChar {
						r += 6
						w += utf8.EncodeRune(b[w:], dec)
						break
					}
					rr = unicode.ReplacementChar
				}
				w += utf8.EncodeRune(b[w:], rr)
			}
		case c == '"', c < ' ':
			return nil, false
		case c < utf8.RuneSelf:
			b[w] = c
			r++
			w++
		default:
			rr, size := utf8.DecodeRune(s[r:])
			r += size
			w += utf8.EncodeRune(b[w:], rr)
		}
	}
	return b[:w], true
}

func QuotedFrom[T any, PT interface {
	*T
	UnmarshalJSONFrom(*jsontext.Decoder) error
}](s *Scanner, dst PT) bool {
	if s.scalar() == 0 {
		return true
	}
	tok := s.data[s.start:s.end]
	dec := jsontext.NewDecoder(bytes.NewReader(tok), stdjson.DefaultOptionsV1())
	err := dst.UnmarshalJSONFrom(dec)
	if errors.Is(err, errors.ErrUnsupported) {
		if dec.InputOffset() != 0 {
			s.save(errors.New("unsupported calls must not read or write any tokens"))
			return true
		}
		s.pos = s.start
		return false
	}
	if err == nil && dec.InputOffset() != int64(len(tok)) {
		s.save(errors.New("must read or write exactly one value"))
		return true
	}
	s.saveMethodError(err)
	return true
}

func QuotedFromPtr[E any, T interface {
	*E
	UnmarshalJSONFrom(*jsontext.Decoder) error
}](s *Scanner, dst *T) bool {
	if s.scalar() == 0 {
		return true
	}
	tok := s.data[s.start:s.end]
	if s.kind == 'n' {
		*dst = nil
		return true
	}
	if *dst == nil {
		*dst = new(E)
	}
	dec := jsontext.NewDecoder(bytes.NewReader(tok), stdjson.DefaultOptionsV1())
	err := (*dst).UnmarshalJSONFrom(dec)
	if errors.Is(err, errors.ErrUnsupported) {
		if dec.InputOffset() != 0 {
			s.save(errors.New("unsupported calls must not read or write any tokens"))
			return true
		}
		s.pos = s.start
		return false
	}
	if err == nil && dec.InputOffset() != int64(len(tok)) {
		s.save(errors.New("must read or write exactly one value"))
		return true
	}
	s.saveMethodError(err)
	if err == nil && bytes.Equal(tok, quotedNullLiteral) {
		*dst = nil
	}
	return true
}

func (s *Scanner) saveMethodError(err error) {
	if ute, ok := err.(*stdjson.UnmarshalTypeError); ok {
		if s.saved == nil {
			s.saved = ute
		}
		return
	}
	s.fatal(err)
}
