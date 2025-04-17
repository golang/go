// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2

package http

import (
	"reflect"
)

func (e http2StreamError) As(target any) bool {
	dst := reflect.ValueOf(target).Elem()
	dstType := dst.Type()
	if dstType.Kind() != reflect.Struct {
		return false
	}
	src := reflect.ValueOf(e)
	srcType := src.Type()
	numField := srcType.NumField()
	if dstType.NumField() != numField {
		return false
	}
	for i := 0; i < numField; i++ {
		sf := srcType.Field(i)
		df := dstType.Field(i)
		if sf.Name != df.Name || !sf.Type.ConvertibleTo(df.Type) {
			return false
		}
	}
	for i := 0; i < numField; i++ {
		df := dst.Field(i)
		df.Set(src.Field(i).Convert(df.Type()))
	}
	return true
}
