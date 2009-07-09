// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes";
	"gob";
	"os";
	"reflect";
	"strings";
	"testing";
	"unsafe";
)

type ET2 struct {
	x string;
}

type ET1 struct {
	a int;
	et2 *ET2;
	next *ET1;
}

func TestBasicEncoder(t *testing.T) {
	b := new(bytes.Buffer);
	enc := NewEncoder(b);
	et1 := new(ET1);
	et1.a = 7;
	et1.et2 = new(ET2);
	enc.Encode(et1);
	if enc.state.err != nil {
		t.Error("encoder fail:", enc.state.err)
	}
}
