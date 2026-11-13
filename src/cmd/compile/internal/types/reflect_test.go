// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"reflect"
	"strings"
	"testing"
)

// This is a test of the compiler's use of the output of NameString and
// the interaction of that compiled result with the reflect package.
// It-would-be-nice to reduce the size of large type strings,
// but however this is done, it must be compatible with existing
// reflect behavior.

type Pair[A, B any] struct {
	A A
	B B
}
type (
	L1 = Pair[int, int]
	L2 = Pair[L1, L1]
	L3 = Pair[L2, L2]
	L4 = Pair[L3, L3]
	L5 = Pair[L4, L4] // Named generic instantiation: t.Sym() != nil, len > 500 bytes
)
type BigUnnamed = struct {
	F0, F1, F2, F3, F4, F5, F6, F7, F8, F9 struct {
		FieldWithDescriptiveName0000000000000000 int
	}
}

func TestCorruptedReflectTypeString(t *testing.T) {
	valT := reflect.TypeOf(L5{})
	ptrT := reflect.TypeOf((*L5)(nil))
	sliceT := reflect.TypeOf([]L5(nil))
	unnamedT := reflect.TypeOf(BigUnnamed{})

	if !strings.HasPrefix(ptrT.String(), "*") {
		t.Errorf("pointer type String() = %q; does not start with '*'", ptrT.String())
	}
	// 2. (*L5), ([]L5), and BigUnnamed all have 64-char SHA-256 hex hashes as their reflection String().
	if want := "*" + valT.String(); ptrT.String() != want {
		t.Errorf("(*L5).String() = %q, want %q", ptrT.String(), want)
	}
	if want := "[]" + valT.String(); sliceT.String() != want {
		t.Errorf("([]L5).String() = %q, want %q", sliceT.String(), want)
	}
	if !strings.HasPrefix(unnamedT.String(), "struct {") {
		t.Errorf("BigUnnamed.String() = %q, want prefix \"struct {\"", unnamedT.String())
	}
}
func TestSliceOfTypeIdentityAndPanic(t *testing.T) {

	compiledSliceT := reflect.TypeOf([]L5(nil))
	runtimeSliceT := reflect.SliceOf(reflect.TypeOf(L5{}))
	if compiledSliceT != runtimeSliceT {
		t.Errorf("reflect.SliceOf(TypeOf(L5{})) != reflect.TypeOf([]L5(nil)):\n  compiled: %s\n  runtime:  %s",
			compiledSliceT.String(), runtimeSliceT.String())
	}

	v := reflect.MakeSlice(runtimeSliceT, 1, 1).Interface()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("type assertion v.([]L5) panicked: %v", r)
		}
	}()
	_ = v.([]L5)
}
func TestMapOfAndChanOfIdentityAndPanic(t *testing.T) {
	elemT := reflect.TypeOf(L5{})
	if reflect.MapOf(reflect.TypeOf(0), elemT) != reflect.TypeOf(map[int]L5(nil)) {
		t.Errorf("reflect.MapOf(int, L5) != reflect.TypeOf(map[int]L5(nil))")
	}
	if reflect.ChanOf(reflect.BothDir, elemT) != reflect.TypeOf((chan L5)(nil)) {
		t.Errorf("reflect.ChanOf(BothDir, L5) != reflect.TypeOf((chan L5)(nil))")
	}
	if reflect.ArrayOf(1, elemT) != reflect.TypeOf([1]L5{}) {
		t.Errorf("reflect.ArrayOf(1, L5) != reflect.TypeOf([1]L5{})")
	}
	mv := reflect.MakeMap(reflect.MapOf(reflect.TypeOf(0), elemT)).Interface()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("type assertion mv.(map[int]L5) panicked: %v", r)
		}
	}()
	_ = mv.(map[int]L5)
}
