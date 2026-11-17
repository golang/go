// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	"encoding"
	stdjson "encoding/json"
	"reflect"
)

const indexedFieldPaths = false

func fallbackStructName(inner, root string) string {
	if inner != "" {
		return inner
	}
	return root
}

// QuotedNull consumes a string containing null under the active profile's
// ,string pointer rule. Other values remain for the typed scalar decoder.
func (s *Scanner) QuotedNull() bool {
	pos := s.pos
	item, st := s.readQuoted()
	if st == quotedContent && item == "null" {
		return true
	}
	s.pos = pos
	return false
}

func (s *Scanner) rejectQuotedBare(typ reflect.Type) {
	s.save(tagUnquoted(typ))
}

func (s *Scanner) rejectQuotedContent(res quotedResult, item string, typ reflect.Type) {
	switch res {
	case qSavedTag:
		s.save(tagMismatch(item, typ))
	case qFatalTag:
		s.fatal(tagMismatch(item, typ))
	case qString:
		s.rejectKind(typ)
	}
}

func quotedIntFallback(item string, bits int) (int64, bool) {
	return 0, false
}

func quotedUintFallback(item string, bits int) (uint64, bool) {
	return 0, false
}

func quotedFloatFallback(item string, bits int) (float64, bool) {
	return 0, false
}

func unquoteStringContent(item string) (string, bool) {
	return unquoteLiteral(item)
}

func QuotedJSON[T stdjson.Unmarshaler](s *Scanner, dst T) {
	item, st := s.readQuoted()
	switch st {
	case quotedBare:
		s.save(tagUnquoted(reflect.TypeOf(dst).Elem()))
	case quotedContent:
		if item == "" {
			s.save(tagMismatch(item, reflect.TypeOf(dst).Elem()))
			return
		}
		s.fatalMethodError(dst.UnmarshalJSON([]byte(item)))
	default:
		if s.kind == 'n' {
			s.fatalMethodError(dst.UnmarshalJSON(s.data[s.start:s.end]))
		}
	}
}

func QuotedJSONPtr[E any, T interface {
	*E
	stdjson.Unmarshaler
}](s *Scanner, dst *T) {
	item, st := s.readQuoted()
	elem := reflect.TypeOf(dst).Elem()
	switch st {
	case quotedBare:
		s.save(tagUnquoted(elem))
	case quotedContent:
		if item == "" {
			s.save(tagMismatch(item, elem))
			return
		}
		if item[0] == 'n' {
			if item == "null" {
				*dst = nil
				return
			}
			s.save(tagMismatch(item, elem))
			return
		}
		if *dst == nil {
			*dst = new(E)
		}
		s.fatalMethodError((*dst).UnmarshalJSON([]byte(item)))
	default:
		if s.kind == 'n' {
			*dst = nil
		}
	}
}

func QuotedText[E any, T interface {
	*E
	encoding.TextUnmarshaler
}](s *Scanner, dst T) {
	item, st := s.readQuoted()
	switch st {
	case quotedBare:
		s.save(tagUnquoted(reflect.TypeFor[E]()))
	case quotedContent:
		if item == "" {
			s.save(tagMismatch(item, reflect.TypeFor[E]()))
			return
		}
		if item[0] == 'n' {
			if item == "null" {
				return
			}
			s.save(tagMismatch(item, reflect.TypeFor[E]()))
			return
		}
		if item[0] != '"' {
			s.save(tagMismatch(item, reflect.TypeFor[E]()))
			return
		}
		unq, ok := unquoteLiteralBytes(item)
		if !ok {
			s.fatal(tagMismatch(item, reflect.TypeFor[E]()))
			return
		}
		s.fatalMethodError(dst.UnmarshalText(unq))
	default:
	}
}

func QuotedTextPtr[E any, T interface {
	*E
	encoding.TextUnmarshaler
}](s *Scanner, dst *T) {
	item, st := s.readQuoted()
	elem := reflect.TypeOf(dst).Elem()
	switch st {
	case quotedBare:
		s.save(tagUnquoted(elem))
	case quotedContent:
		if item == "" {
			s.save(tagMismatch(item, elem))
			return
		}
		if item[0] == 'n' {
			if item == "null" {
				*dst = nil
				return
			}
			s.save(tagMismatch(item, elem))
			return
		}
		if *dst == nil {
			*dst = new(E)
		}
		if item[0] != '"' {
			s.save(tagMismatch(item, elem))
			return
		}
		unq, ok := unquoteLiteralBytes(item)
		if !ok {
			s.fatal(tagMismatch(item, elem))
			return
		}
		s.fatalMethodError((*dst).UnmarshalText(unq))
	default:
		if s.kind == 'n' {
			*dst = nil
		}
	}
}

func QuotedFrom[T any, PT interface{ *T }](s *Scanner, dst PT) bool {
	return false
}

func QuotedFromPtr[E any, T interface{ *E }](s *Scanner, dst *T) bool {
	return false
}

func (s *Scanner) fatalMethodError(err error) {
	if ute, ok := err.(*stdjson.UnmarshalTypeError); ok {
		ute.Struct = s.structName()
		if ute.Field == "" {
			ute.Field = detach(s.key)
		} else {
			ute.Field = s.key + "." + ute.Field
		}
	}
	s.fatal(err)
}
