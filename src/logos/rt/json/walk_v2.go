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
	"unsafe"
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

func pointerVisit[P ~*T, T any](p P) walkVisit {
	return walkVisit{typ: typeToken[P](), ptr: unsafe.Pointer(p)}
}

func sliceVisit[S ~[]E, E any](v S) walkVisit {
	return walkVisit{typ: typeToken[S](), ptr: sliceData(v), len: len(v)}
}

func mapVisit[M ~map[K]V, K comparable, V any](m M) walkVisit {
	return walkVisit{typ: typeToken[M](), ptr: mapData(m)}
}

func valueVisit(v reflect.Value) walkVisit {
	key := walkVisit{typ: valueToken(v.Type()), ptr: v.UnsafePointer()}
	if v.Kind() == reflect.Slice {
		key.len = v.Len()
	}
	return key
}
