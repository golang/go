// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	stdjson "encoding/json"
	"reflect"
)

func (s *Walk) checkDepth(offset int64) error {
	return nil
}

var walkMethodTypes = []reflect.Type{marshalerType, textMarshalerType}

func (s *Walk) checkMapKeys(rv reflect.Value) error {
	if !walkMapKeySupported(rv.Type().Key()) {
		return &stdjson.UnsupportedTypeError{Type: rv.Type()}
	}
	return nil
}

func visitKey(v reflect.Value) any {
	switch v.Kind() {
	case reflect.Pointer:
		return v.Interface()
	case reflect.Slice:
		return struct {
			ptr any
			len int
		}{v.UnsafePointer(), v.Len()}
	case reflect.Map:
		return v.UnsafePointer()
	}
	return nil
}
