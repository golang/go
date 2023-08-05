// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"testing"
)

var output int

type Object struct {
	Val int
}

func (o *Object) Initialize() *Object {
	o.Val = 5
	return o
}

func (o *Object) Update() *Object {
	o.Val = o.Val + 1
	return o
}

func TestAutotmpLoopDepth(t *testing.T) {
	f := func() {
		for i := 0; i < 10; i++ {
			var obj Object
			obj.Initialize().Update()
			output = obj.Val
		}
	}
	if n := testing.AllocsPerRun(10, f); n > 0 {
		t.Error("obj moved to heap")
	}
}
