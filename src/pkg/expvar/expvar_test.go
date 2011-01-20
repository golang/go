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

func TestFloat(t *testing.T) {
	reqs := NewFloat("requests-float")
	if reqs.f != 0.0 {
		t.Errorf("reqs.f = %v, want 0", reqs.f)
	}
	if reqs != Get("requests-float").(*Float) {
		t.Errorf("Get() failed.")
	}

	reqs.Add(1.5)
	reqs.Add(1.25)
	if reqs.f != 2.75 {
		t.Errorf("reqs.f = %v, want 2.75", reqs.f)
	}

	if s := reqs.String(); s != "2.75" {
		t.Errorf("reqs.String() = %q, want \"4.64\"", s)
	}

	reqs.Add(-2)
	if reqs.f != 0.75 {
		t.Errorf("reqs.f = %v, want 0.75", reqs.f)
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
	colours.AddFloat("green", 4.125)
	if x := colours.m["red"].(*Int).i; x != 3 {
		t.Errorf("colours.m[\"red\"] = %v, want 3", x)
	}
	if x := colours.m["blue"].(*Int).i; x != 4 {
		t.Errorf("colours.m[\"blue\"] = %v, want 4", x)
	}
	if x := colours.m["green"].(*Float).f; x != 4.125 {
		t.Errorf("colours.m[\"green\"] = %v, want 3.14", x)
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
	x := int64(4)
	ix := IntFunc(func() int64 { return x })
	if s := ix.String(); s != "4" {
		t.Errorf("ix.String() = %v, want 4", s)
	}

	x++
	if s := ix.String(); s != "5" {
		t.Errorf("ix.String() = %v, want 5", s)
	}
}

func TestFloatFunc(t *testing.T) {
	x := 8.5
	ix := FloatFunc(func() float64 { return x })
	if s := ix.String(); s != "8.5" {
		t.Errorf("ix.String() = %v, want 3.14", s)
	}

	x -= 1.25
	if s := ix.String(); s != "7.25" {
		t.Errorf("ix.String() = %v, want 4.34", s)
	}
}

func TestStringFunc(t *testing.T) {
	x := "hello"
	sx := StringFunc(func() string { return x })
	if s, exp := sx.String(), `"hello"`; s != exp {
		t.Errorf(`sx.String() = %q, want %q`, s, exp)
	}

	x = "goodbye"
	if s, exp := sx.String(), `"goodbye"`; s != exp {
		t.Errorf(`sx.String() = %q, want %q`, s, exp)
	}
}
