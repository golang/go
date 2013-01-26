// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package expvar

import (
	"encoding/json"
	"testing"
)

// RemoveAll removes all exported variables.
// This is for tests only.
func RemoveAll() {
	mutex.Lock()
	defer mutex.Unlock()
	vars = make(map[string]Var)
}

func TestInt(t *testing.T) {
	RemoveAll()
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
	RemoveAll()
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
	RemoveAll()
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
	RemoveAll()
	colors := NewMap("bike-shed-colors")

	colors.Add("red", 1)
	colors.Add("red", 2)
	colors.Add("blue", 4)
	colors.AddFloat("green", 4.125)
	if x := colors.m["red"].(*Int).i; x != 3 {
		t.Errorf("colors.m[\"red\"] = %v, want 3", x)
	}
	if x := colors.m["blue"].(*Int).i; x != 4 {
		t.Errorf("colors.m[\"blue\"] = %v, want 4", x)
	}
	if x := colors.m["green"].(*Float).f; x != 4.125 {
		t.Errorf("colors.m[\"green\"] = %v, want 3.14", x)
	}

	// colors.String() should be '{"red":3, "blue":4}',
	// though the order of red and blue could vary.
	s := colors.String()
	var j interface{}
	err := json.Unmarshal([]byte(s), &j)
	if err != nil {
		t.Errorf("colors.String() isn't valid JSON: %v", err)
	}
	m, ok := j.(map[string]interface{})
	if !ok {
		t.Error("colors.String() didn't produce a map.")
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

func TestFunc(t *testing.T) {
	RemoveAll()
	var x interface{} = []string{"a", "b"}
	f := Func(func() interface{} { return x })
	if s, exp := f.String(), `["a","b"]`; s != exp {
		t.Errorf(`f.String() = %q, want %q`, s, exp)
	}

	x = 17
	if s, exp := f.String(), `17`; s != exp {
		t.Errorf(`f.String() = %q, want %q`, s, exp)
	}
}
