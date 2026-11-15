// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"encoding/json/jsontext"
	"errors"
	"io"
	"reflect"
	"strconv"
	"strings"
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
