// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

var CheckFunc = checkFunc
var PrintFunc = printFunc
var Opt = opt
var Deadcode = deadcode

func testConfig(t *testing.T) *Config {
	return NewConfig("amd64", DummyFrontend{t})
}

// DummyFrontend is a test-only frontend.
// It assumes 64 bit integers and pointers.
type DummyFrontend struct {
	t testing.TB
}

func (DummyFrontend) StringData(s string) interface{} {
	return nil
}

func (d DummyFrontend) Logf(msg string, args ...interface{})           { d.t.Logf(msg, args...) }
func (d DummyFrontend) Fatalf(msg string, args ...interface{})         { d.t.Fatalf(msg, args...) }
func (d DummyFrontend) Unimplementedf(msg string, args ...interface{}) { d.t.Fatalf(msg, args...) }

func (d DummyFrontend) TypeBool() Type    { return TypeBool }
func (d DummyFrontend) TypeInt8() Type    { return TypeInt8 }
func (d DummyFrontend) TypeInt16() Type   { return TypeInt16 }
func (d DummyFrontend) TypeInt32() Type   { return TypeInt32 }
func (d DummyFrontend) TypeInt64() Type   { return TypeInt64 }
func (d DummyFrontend) TypeUInt8() Type   { return TypeUInt8 }
func (d DummyFrontend) TypeUInt16() Type  { return TypeUInt16 }
func (d DummyFrontend) TypeUInt32() Type  { return TypeUInt32 }
func (d DummyFrontend) TypeUInt64() Type  { return TypeUInt64 }
func (d DummyFrontend) TypeInt() Type     { return TypeInt64 }
func (d DummyFrontend) TypeUintptr() Type { return TypeUInt64 }
func (d DummyFrontend) TypeString() Type  { panic("unimplemented") }
func (d DummyFrontend) TypeBytePtr() Type { return TypeBytePtr }
