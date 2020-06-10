// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
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
