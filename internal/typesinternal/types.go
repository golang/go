// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typesinternal provides access to internal go/types APIs that are not
// yet exported.
package typesinternal

import (
	"go/token"
	"go/types"
	"reflect"
	"unsafe"
)

func SetUsesCgo(conf *types.Config) bool {
	v := reflect.ValueOf(conf).Elem()

	f := v.FieldByName("go115UsesCgo")
	if !f.IsValid() {
		f = v.FieldByName("UsesCgo")
		if !f.IsValid() {
			return false
		}
	}

	addr := unsafe.Pointer(f.UnsafeAddr())
	*(*bool)(addr) = true

	return true
}

func ReadGo116ErrorData(terr types.Error) (ErrorCode, token.Pos, token.Pos, bool) {
	var data [3]int
	// By coincidence all of these fields are ints, which simplifies things.
	v := reflect.ValueOf(terr)
	for i, name := range []string{"go116code", "go116start", "go116end"} {
		f := v.FieldByName(name)
		if !f.IsValid() {
			return 0, 0, 0, false
		}
		data[i] = int(f.Int())
	}
	return ErrorCode(data[0]), token.Pos(data[1]), token.Pos(data[2]), true
}
