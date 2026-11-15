// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"math"
	"reflect"
	"strconv"
	"unsafe"
)

type readStatus uint8

const (
	readNone readStatus = iota
	readOK
	readMismatch
	readRange
)

type quotedStatus uint8

const (
	quotedNone quotedStatus = iota
	quotedContent
	quotedBare
)

type quotedResult uint8

const (
	qNoop quotedResult = iota
	qSet
	qSavedTag
	qFatalTag
	qString
	qRange
)

type tagError struct {
	msg string
}

func (e *tagError) Error() string {
	return e.msg
}

func tagMismatch(item string, typ reflect.Type) error {
	return &tagError{"json: invalid use of ,string struct tag, trying to unmarshal " + strconv.Quote(item) + " into " + typ.String()}
}

func tagUnquoted(typ reflect.Type) error {
	return &tagError{"json: invalid use of ,string struct tag, trying to unmarshal unquoted value into " + typ.String()}
}

func parseMagnitude(digits string) (uint64, bool) {
	if digits == "" {
		return 0, false
	}
	var mag uint64
	for i := range len(digits) {
		d := digits[i] - '0'
		if d > 9 || mag > (math.MaxUint64-uint64(d))/10 {
			return 0, false
		}
		mag = mag*10 + uint64(d)
	}
	return mag, true
}

func parseInt(lit string, bits int) (int64, bool) {
	neg := lit != "" && lit[0] == '-'
	if neg {
		lit = lit[1:]
	}
	mag, ok := parseMagnitude(lit)
	if !ok {
		return 0, false
	}
	limit := uint64(1) << (bits - 1)
	if neg {
		return int64(-mag), mag <= limit
	}
	return int64(mag), mag < limit
}

func parseUint(lit string, bits int) (uint64, bool) {
	mag, ok := parseMagnitude(lit)
	if !ok {
		return 0, false
	}
	return mag, bits == 64 || mag <= uint64(1)<<bits-1
}

func (s *Scanner) literal() string {
	return unsafeString(s.data[s.start:s.end])
}

func (s *Scanner) readBool() (bool, readStatus) {
	switch s.scalar() {
	case 0, 'n':
		return false, readNone
	case 't':
		return true, readOK
	case 'f':
		return false, readOK
	}
	return false, readMismatch
}

func (s *Scanner) readInt(bits int) (int64, readStatus) {
	switch s.scalar() {
	case 0, 'n':
		return 0, readNone
	case '0':
		if v, ok := parseInt(s.literal(), bits); ok {
			return v, readOK
		}
		return 0, readRange
	}
	return 0, readMismatch
}

func (s *Scanner) readUint(bits int) (uint64, readStatus) {
	switch s.scalar() {
	case 0, 'n':
		return 0, readNone
	case '0':
		if v, ok := parseUint(s.literal(), bits); ok {
			return v, readOK
		}
		return 0, readRange
	}
	return 0, readMismatch
}

func (s *Scanner) readFloat(bits int) (float64, readStatus) {
	switch s.scalar() {
	case 0, 'n':
		return 0, readNone
	case '0':
		if v, err := strconv.ParseFloat(s.literal(), bits); err == nil {
			return v, readOK
		}
		return 0, readRange
	}
	return 0, readMismatch
}

func (s *Scanner) readString() (string, readStatus) {
	switch s.scalar() {
	case 0, 'n':
		return "", readNone
	case '"':
		return s.stringBody(), readOK
	}
	return "", readMismatch
}

func reject[T any](s *Scanner, st readStatus) {
	switch st {
	case readMismatch:
		s.rejectKind(reflect.TypeFor[T]())
	case readRange:
		s.rejectNumber(s.literal(), reflect.TypeFor[T]())
	}
}

// Bool decodes the JSON value at the cursor into *dst. null leaves *dst
// unchanged. A value that is not a boolean saves one *UnmarshalTypeError
// and is skipped, like encoding/json; decoding continues. Syntax errors are
// fatal.
func Bool[T ~bool](s *Scanner, dst *T) {
	v, st := s.readBool()
	if st == readOK {
		*dst = T(v)
		return
	}
	reject[T](s, st)
}

// Int decodes the JSON value at the cursor into *dst. null leaves *dst
// unchanged. A kind or range mismatch saves one *UnmarshalTypeError (Value
// "string", "number 300", "number 1.5", "bool", "array" or "object", Type
// the Go type of *dst, Offset just after the value, Struct and Field from
// the Scanner) and skips the value, like encoding/json. The bit width is
// unsafe.Sizeof(*dst)*8.
func Int[T ~int | ~int8 | ~int16 | ~int32 | ~int64](s *Scanner, dst *T) {
	v, st := s.readInt(int(unsafe.Sizeof(*dst)) * 8)
	if st == readOK {
		*dst = T(v)
		return
	}
	reject[T](s, st)
}

// Uint decodes the JSON value at the cursor into *dst like Int, for
// unsigned types.
func Uint[T ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr](s *Scanner, dst *T) {
	v, st := s.readUint(int(unsafe.Sizeof(*dst)) * 8)
	if st == readOK {
		*dst = T(v)
		return
	}
	reject[T](s, st)
}

// Float decodes the JSON value at the cursor into *dst like Int. A float32
// destination parses with 32 bit precision, and a number out of range
// saves "number 1e999".
func Float[T ~float32 | ~float64](s *Scanner, dst *T) {
	v, st := s.readFloat(int(unsafe.Sizeof(*dst)) * 8)
	if st == readOK {
		*dst = T(v)
		return
	}
	reject[T](s, st)
}

// String decodes the JSON value at the cursor into *dst like Int. It
// handles every escape, including \uXXXX surrogate pairs; a lone surrogate
// and an invalid UTF-8 byte become U+FFFD, and a control character inside
// the string is a syntax error.
func String[T ~string](s *Scanner, dst *T) {
	v, st := s.readString()
	if st == readOK {
		*dst = T(v)
		return
	}
	reject[T](s, st)
}

func (s *Scanner) readQuoted() (string, quotedStatus) {
	switch s.scalar() {
	case 0, 'n':
		return "", quotedNone
	case '"':
		return s.stringBody(), quotedContent
	}
	return "", quotedBare
}

func quotedOther(item string) quotedResult {
	if item == "" {
		return qSavedTag
	}
	switch item[0] {
	case 'n', 't', 'f':
		return qSavedTag
	case '"':
		if _, ok := unquoteStringContent(item); ok {
			return qString
		}
	}
	return qFatalTag
}

func quotedBoolValue(item string) (bool, quotedResult) {
	switch item {
	case "true":
		return true, qSet
	case "false":
		return false, qSet
	case "null":
		return false, qNoop
	}
	return false, quotedOther(item)
}

func quotedIntValue(item string, bits int) (int64, quotedResult) {
	if item == "null" {
		return 0, qNoop
	}
	if item == "" || !isNumberStart(item[0]) {
		if v, ok := quotedIntFallback(item, bits); ok {
			return v, qSet
		}
		return 0, quotedOther(item)
	}
	if v, ok := parseInt(item, bits); ok {
		return v, qSet
	}
	return 0, qRange
}

func quotedUintValue(item string, bits int) (uint64, quotedResult) {
	if item == "null" {
		return 0, qNoop
	}
	if item == "" || !isNumberStart(item[0]) {
		if v, ok := quotedUintFallback(item, bits); ok {
			return v, qSet
		}
		return 0, quotedOther(item)
	}
	if v, ok := parseUint(item, bits); ok {
		return v, qSet
	}
	return 0, qRange
}

func quotedFloatValue(item string, bits int) (float64, quotedResult) {
	if item == "null" {
		return 0, qNoop
	}
	if item == "" || !isNumberStart(item[0]) {
		if v, ok := quotedFloatFallback(item, bits); ok {
			return v, qSet
		}
		return 0, quotedOther(item)
	}
	if v, err := strconv.ParseFloat(item, bits); err == nil {
		return v, qSet
	}
	return 0, qRange
}

func quotedStringValue(item string) (string, quotedResult) {
	if item == "null" {
		return "", qNoop
	}
	if item != "" && item[0] == '"' {
		if v, ok := unquoteStringContent(item); ok {
			return v, qSet
		}
		return "", qFatalTag
	}
	return "", quotedOther(item)
}

func rejectQuoted[T any](s *Scanner, res quotedResult, item string) {
	switch res {
	case qSavedTag, qFatalTag, qString:
		s.rejectQuotedContent(res, item, reflect.TypeFor[T]())
	case qRange:
		s.rejectNumber(item, reflect.TypeFor[T]())
	}
}

func readQuotedAs[T any](s *Scanner, dst *T, parse func(string) (T, quotedResult)) {
	item, st := s.readQuoted()
	switch st {
	case quotedBare:
		s.rejectQuotedBare(reflect.TypeFor[T]())
	case quotedContent:
		v, res := parse(item)
		if res == qSet {
			*dst = v
			return
		}
		rejectQuoted[T](s, res, item)
	}
}

// BoolQ decodes a bool field that carries the ,string tag option: the
// value must be a JSON string whose content is the literal true or false.
// null leaves *dst unchanged. A non-string value saves
// `json: invalid use of ,string struct tag, trying to unmarshal unquoted
// value into <type>`, and the other malformed contents follow the rules
// of encoding/json for the option.
func BoolQ[T ~bool](s *Scanner, dst *T) {
	readQuotedAs(s, dst, func(item string) (T, quotedResult) {
		v, res := quotedBoolValue(item)
		return T(v), res
	})
}

// IntQ is BoolQ for an integer field, whose string content is a decimal
// integer.
func IntQ[T ~int | ~int8 | ~int16 | ~int32 | ~int64](s *Scanner, dst *T) {
	bits := int(unsafe.Sizeof(*dst)) * 8
	readQuotedAs(s, dst, func(item string) (T, quotedResult) {
		v, res := quotedIntValue(item, bits)
		return T(v), res
	})
}

// UintQ is BoolQ for an unsigned integer field.
func UintQ[T ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr](s *Scanner, dst *T) {
	bits := int(unsafe.Sizeof(*dst)) * 8
	readQuotedAs(s, dst, func(item string) (T, quotedResult) {
		v, res := quotedUintValue(item, bits)
		return T(v), res
	})
}

// FloatQ is BoolQ for a floating-point field.
func FloatQ[T ~float32 | ~float64](s *Scanner, dst *T) {
	bits := int(unsafe.Sizeof(*dst)) * 8
	readQuotedAs(s, dst, func(item string) (T, quotedResult) {
		v, res := quotedFloatValue(item, bits)
		return T(v), res
	})
}

// StringQ is BoolQ for a string field: the content of the JSON string must
// itself be a valid JSON string literal, which is unquoted into *dst.
func StringQ[T ~string](s *Scanner, dst *T) {
	readQuotedAs(s, dst, func(item string) (T, quotedResult) {
		v, res := quotedStringValue(item)
		return T(v), res
	})
}
