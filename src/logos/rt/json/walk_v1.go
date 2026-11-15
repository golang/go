// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	"reflect"
	"unsafe"
)

func (s *Walk) checkDepth(offset int64) error {
	return nil
}

var walkMethodTypes = []reflect.Type{marshalerType, textMarshalerType}

func pointerVisit[P ~*T, T any](p P) walkVisit {
	return walkVisit{typ: typeToken[P](), ptr: unsafe.Pointer(p)}
}

func sliceVisit[S ~[]E, E any](v S) walkVisit {
	return walkVisit{ptr: sliceData(v), len: len(v)}
}

func mapVisit[M ~map[K]V, K comparable, V any](m M) walkVisit {
	return walkVisit{ptr: mapData(m)}
}

func valueVisit(v reflect.Value) walkVisit {
	switch v.Kind() {
	case reflect.Pointer:
		return walkVisit{typ: valueToken(v.Type()), ptr: v.UnsafePointer()}
	case reflect.Slice:
		return walkVisit{ptr: v.UnsafePointer(), len: v.Len()}
	}
	return walkVisit{ptr: v.UnsafePointer()}
}
