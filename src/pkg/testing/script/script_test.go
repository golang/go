// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"testing"
)

func TestNoop(t *testing.T) {
	err := Perform(0, nil)
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func TestSimple(t *testing.T) {
	c := make(chan int)
	defer close(c)

	a := NewEvent("send", nil, Send{c, 1})
	b := NewEvent("recv", []*Event{a}, Recv{c, 1})

	err := Perform(0, []*Event{a, b})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func TestFail(t *testing.T) {
	c := make(chan int)
	defer close(c)

	a := NewEvent("send", nil, Send{c, 2})
	b := NewEvent("recv", []*Event{a}, Recv{c, 1})

	err := Perform(0, []*Event{a, b})
	if err == nil {
		t.Errorf("Failed to get expected error")
	} else if _, ok := err.(ReceivedUnexpected); !ok {
		t.Errorf("Error returned was of the wrong type: %s", err)
	}
}

func TestClose(t *testing.T) {
	c := make(chan int)

	a := NewEvent("close", nil, Close{c})
	b := NewEvent("closed", []*Event{a}, Closed{c})

	err := Perform(0, []*Event{a, b})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func matchOne(v interface{}) bool {
	if i, ok := v.(int); ok && i == 1 {
		return true
	}
	return false
}

func TestRecvMatch(t *testing.T) {
	c := make(chan int)

	a := NewEvent("send", nil, Send{c, 1})
	b := NewEvent("recv", []*Event{a}, RecvMatch{c, matchOne})

	err := Perform(0, []*Event{a, b})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}
