// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	stdjson "encoding/json"
	jsonv2 "encoding/json/v2"
	"reflect"
	"strings"
)

const maxWalkDepth = 10000

var walkMethodTypes = []reflect.Type{
	reflect.TypeFor[jsonv2.MarshalerTo](),
	marshalerType,
	textAppenderType,
	textMarshalerType,
}

func newWalkDepthError(written int64) *stdjson.SyntaxError {
	var v any
	err := stdjson.Unmarshal([]byte(strings.Repeat("[", maxWalkDepth+1)), &v)
	e := *err.(*stdjson.SyntaxError)
	e.Offset = written + 1
	return &e
}

func (s *Walk) checkDepth(offset int64) error {
	if s.depth >= maxWalkDepth {
		s.depthErr = newWalkDepthError(offset)
		return s.depthErr
	}
	return nil
}

func (s *Walk) checkMapKeys(rv reflect.Value) error {
	if rv.IsNil() {
		return nil
	}
	if !walkMapKeySupported(rv.Type().Key()) {
		return &stdjson.UnsupportedTypeError{Type: rv.Type()}
	}
	return nil
}

func sliceWalkLen(v reflect.Value) int {
	if v.Kind() == reflect.Slice {
		return v.Len()
	}
	return 0
}

func visitKey(v reflect.Value) any {
	return walkVisit{v.Type(), v.UnsafePointer(), sliceWalkLen(v)}
}
