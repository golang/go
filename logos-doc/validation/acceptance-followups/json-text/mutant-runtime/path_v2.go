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
	"reflect"
	"strconv"
)

const indexedFieldPaths = true

func fallbackStructName(_ string, root string) string {
	return root
}

// QuotedNull consumes the exact token "null" under JSONv2's legacy
// ,string pointer rule. Escaped spellings leave the pointer unchanged.
func (s *Scanner) QuotedNull() bool {
	i := skipSpace(s.data, s.pos)
	end := i + len(`"null"`)
	if end > len(s.data) || string(s.data[i:end]) != `"null"` {
		return false
	}
	if end < len(s.data) && !isDelimiter(s.data[end]) {
		return false
	}
	s.pos = end
	return true
}

func stringContentError(item string) error {
	// SyntaxError's message is private. Let encoding/json construct the
	// public error, including its type and offset, on this error path.
	var dst struct {
		V string `json:"v,string"`
	}
	raw := AppendEscaped([]byte(`{"v":"`), item)
	raw = append(raw, '"', '}')
	err := stdjson.Unmarshal(raw, &dst)
	if err == nil {
		return nil
	}
	if ute, ok := err.(*stdjson.UnmarshalTypeError); ok {
		return ute.Err
	}
	return err
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
		if unq, ok := unquoteLiteralBytes(string(tok)); ok {
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
		if unq, err := jsontext.AppendUnquote(nil, tok); err == nil {
			err = rejectMethodUnsupported((*dst).UnmarshalText(unq), "UnmarshalText method")
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
