// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package expvar

import (
	"json"
	"testing"
)

func TestInt(t *testing.T) {
	reqs := NewInt("requests")
	if reqs.i != 0 {
		t.Errorf("reqs.i = %v, want 0", reqs.i)
	}
	if reqs != Get("requests").(*Int) {
		t.Errorf("Get() failed.")
	}

	reqs.Add(1)
	reqs.Add(3)
	if reqs.i != 4 {
		t.Errorf("reqs.i = %v, want 4", reqs.i)
	}

	if s := reqs.String(); s != "4" {
		t.Errorf("reqs.String() = %q, want \"4\"", s)
	}

	reqs.Set(-2)
	if reqs.i != -2 {
		t.Errorf("reqs.i = %v, want -2", reqs.i)
	}
}

func TestString(t *testing.T) {
	name := NewString("my-name")
	if name.s != "" {
		t.Errorf("name.s = %q, want \"\"", name.s)
	}

	name.Set("Mike")
	if name.s != "Mike" {
		t.Errorf("name.s = %q, want \"Mike\"", name.s)
	}

	if s := name.String(); s != "\"Mike\"" {
		t.Errorf("reqs.String() = %q, want \"\"Mike\"\"", s)
	}
}

func TestMapCounter(t *testing.T) {
	colours := NewMap("bike-shed-colours")

	colours.Add("red", 1)
	colours.Add("red", 2)
	colours.Add("blue", 4)
	if x := colours.m["red"].(*Int).i; x != 3 {
		t.Errorf("colours.m[\"red\"] = %v, want 3", x)
	}
	if x := colours.m["blue"].(*Int).i; x != 4 {
		t.Errorf("colours.m[\"blue\"] = %v, want 4", x)
	}

	// colours.String() should be '{"red":3, "blue":4}',
	// though the order of red and blue could vary.
	s := colours.String()
	var j interface{}
	err := json.Unmarshal([]byte(s), &j)
	if err != nil {
		t.Errorf("colours.String() isn't valid JSON: %v", err)
	}
	m, ok := j.(map[string]interface{})
	if !ok {
		t.Error("colours.String() didn't produce a map.")
	}
	red := m["red"]
	x, ok := red.(float64)
	if !ok {
		t.Error("red.Kind() is not a number.")
	}
	if x != 3 {
		t.Errorf("red = %v, want 3", x)
	}
}

func TestIntFunc(t *testing.T) {
	x := int(4)
	ix := IntFunc(func() int64 { return int64(x) })
	if s := ix.String(); s != "4" {
		t.Errorf("ix.String() = %v, want 4", s)
	}

	x++
	if s := ix.String(); s != "5" {
		t.Errorf("ix.String() = %v, want 5", s)
	}
}
