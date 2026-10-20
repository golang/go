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

// This file holds the panic value formatter of PanicError. logos/rt may
// import only context, errors, reflect, unsafe, and the packages below
// them (go/build deps_test.go), so it cannot use fmt or runtime/debug:
// panicValueText spells a value the way fmt's %v does for the kinds a
// panic usually carries, and stackTrace is the loop runtime/debug.Stack
// runs.
//
// Spelling, in the order fmt applies it:
//   - nil is `<nil>`;
//   - a value with an Error method is its Error text, and otherwise a
//     value with a String method is its String text, so error wins over
//     Stringer as in fmt;
//   - a bool, any integer kind (uintptr included), a float32 or float64
//     (strconv 'g' with the shortest digits, as fmt's %v), or a string is
//     spelled by its kind, so named types print as their underlying value;
//   - every other kind, complex numbers included, prints as the runtime
//     prints a panic value it cannot format: `(T) 0xADDR`, with T the
//     dynamic type and ADDR the data word of the interface holding the
//     value.
//
// Differences from fmt: fmt.Formatter and fmt.GoStringer are not
// consulted, and a panic inside an Error or String method propagates
// instead of being reported as `%!v(PANIC=...)`.

// panicValueText spells v as described above.
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

// kindText spells v by its reflect kind, or opaquely when the kind has
// no %v spelling here.
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

// opaqueText spells v as `(T) 0xADDR`, the runtime's spelling of a panic
// value of a kind it cannot format. The address is the data word of the
// interface that holds v, read through the layout the runtime defines:
// the type word, then the data word.
func opaqueText(v any) string {
	data := (*[2]unsafe.Pointer)(unsafe.Pointer(&v))[1]
	return "(" + reflect.TypeOf(v).String() + ") 0x" + strconv.FormatUint(uint64(uintptr(data)), 16)
}

// stackTrace returns the stack of the calling goroutine in a buffer that
// doubles until the trace fits, the loop runtime/debug.Stack runs. Called
// from a deferred function it includes the frames of the panic.
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
