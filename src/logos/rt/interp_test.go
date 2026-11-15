// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt_test

import (
	"fmt"
	"testing"

	"logos/rt"
)

type interpPtr struct{ text string }

func (p *interpPtr) Error() string {
	if p.text == "" {
		panic("boom")
	}
	return p.text
}

func (p *interpPtr) String() string {
	if p.text == "" {
		panic("boom")
	}
	return p.text
}

type interpVal struct{ boom bool }

func (v interpVal) Error() string {
	if v.boom {
		panic("boom")
	}
	return "val"
}

func (v interpVal) String() string {
	if v.boom {
		panic("boom")
	}
	return "val"
}

const interpPanicError = "%!v(PANIC=Error method: boom)"

const interpPanicString = "%!v(PANIC=String method: boom)"

func TestInterpErrorHelpers(t *testing.T) {
	var nilPtr *interpPtr
	cases := []struct{ name, got, want string }{
		{"interface-nil", rt.InterpError(nil, fmt.Sprint), "<nil>"},
		{"interface-typed-nil", rt.InterpError(nilPtr, fmt.Sprint), "<nil>"},
		{"interface-panic", rt.InterpError(&interpPtr{}, fmt.Sprint), interpPanicError},
		{"interface-ok", rt.InterpError(&interpPtr{text: "ok"}, fmt.Sprint), "ok"},
		{"value-panic", rt.InterpErrorValue(interpVal{boom: true}, fmt.Sprint), interpPanicError},
		{"value-ok", rt.InterpErrorValue(interpVal{}, fmt.Sprint), "val"},
		{"pointer-nil", rt.InterpErrorPtr(nilPtr, fmt.Sprint), "<nil>"},
		{"pointer-panic", rt.InterpErrorPtr(&interpPtr{}, fmt.Sprint), interpPanicError},
		{"pointer-ok", rt.InterpErrorPtr(&interpPtr{text: "ok"}, fmt.Sprint), "ok"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, c.got, c.want)
		}
	}
}

func TestInterpStringerHelpers(t *testing.T) {
	var nilPtr *interpPtr
	cases := []struct{ name, got, want string }{
		{"interface-nil", rt.InterpStringer(nil, fmt.Sprint), "<nil>"},
		{"interface-typed-nil", rt.InterpStringer(nilPtr, fmt.Sprint), "<nil>"},
		{"interface-panic", rt.InterpStringer(&interpPtr{}, fmt.Sprint), interpPanicString},
		{"interface-ok", rt.InterpStringer(&interpPtr{text: "ok"}, fmt.Sprint), "ok"},
		{"value-panic", rt.InterpStringerValue(interpVal{boom: true}, fmt.Sprint), interpPanicString},
		{"value-ok", rt.InterpStringerValue(interpVal{}, fmt.Sprint), "val"},
		{"pointer-nil", rt.InterpStringerPtr(nilPtr, fmt.Sprint), "<nil>"},
		{"pointer-panic", rt.InterpStringerPtr(&interpPtr{}, fmt.Sprint), interpPanicString},
		{"pointer-ok", rt.InterpStringerPtr(&interpPtr{text: "ok"}, fmt.Sprint), "ok"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, c.got, c.want)
		}
	}
}
