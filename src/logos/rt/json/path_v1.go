// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import "reflect"

const indexedFieldPaths = false

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
