// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package callback

import (
	"testing"
)

func TestCall(t *testing.T) {
	c := NewCaller()
	cb := NewCallback()

	c.SetCallback(cb)
	s := c.Call()
	if s != "Callback::run" {
		t.Errorf("unexpected string from Call: %q", s)
	}
	c.DelCallback()
}

func TestCallback(t *testing.T) {
	c := NewCaller()
	cb := NewDirectorCallback(&GoCallback{})
	c.SetCallback(cb)
	s := c.Call()
	if s != "GoCallback.Run" {
		t.Errorf("unexpected string from Call with callback: %q", s)
	}
	c.DelCallback()
	DeleteDirectorCallback(cb)
}
