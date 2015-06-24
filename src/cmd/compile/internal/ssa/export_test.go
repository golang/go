// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

var CheckFunc = checkFunc
var PrintFunc = printFunc
var Opt = opt
var Deadcode = deadcode

type DummyFrontend struct {
	t *testing.T
}

func (DummyFrontend) StringSym(s string) interface{} {
	return nil
}

func (d DummyFrontend) Logf(msg string, args ...interface{})           { d.t.Logf(msg, args...) }
func (d DummyFrontend) Fatalf(msg string, args ...interface{})         { d.t.Fatalf(msg, args...) }
func (d DummyFrontend) Unimplementedf(msg string, args ...interface{}) { d.t.Fatalf(msg, args...) }
