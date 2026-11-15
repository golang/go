// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"reflect"
	"runtime"
	"strconv"
	"unsafe"
)

func panicValueText(v any) string {
	switch x := v.(type) {
	case nil:
		return "<nil>"
	case error:
		return x.Error()
	case interface{ String() string }:
		return x.String()
	}
	return kindText(v)
}

func kindText(v any) string {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Bool:
		return strconv.FormatBool(rv.Bool())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(rv.Int(), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(rv.Uint(), 10)
	case reflect.Float32:
		return strconv.FormatFloat(rv.Float(), 'g', -1, 32)
	case reflect.Float64:
		return strconv.FormatFloat(rv.Float(), 'g', -1, 64)
	case reflect.String:
		return rv.String()
	}
	return opaqueText(v)
}

func opaqueText(v any) string {
	data := (*[2]unsafe.Pointer)(unsafe.Pointer(&v))[1]
	return "(" + reflect.TypeOf(v).String() + ") 0x" + strconv.FormatUint(uint64(uintptr(data)), 16)
}

func stackTrace() []byte {
	buf := make([]byte, 1024)
	for {
		n := runtime.Stack(buf, false)
		if n < len(buf) {
			return buf[:n]
		}
		buf = make([]byte, 2*len(buf))
	}
}
