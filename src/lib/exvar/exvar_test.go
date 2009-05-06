// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exvar

import (
	"exvar";
	"fmt";
	"json";
	"testing";
)

func TestInt(t *testing.T) {
	reqs := NewInt("requests");
	if reqs.i != 0 {
		t.Errorf("reqs.i = %v, want 4", reqs.i)
	}
	if reqs != Get("requests").(*Int) {
		t.Errorf("Get() failed.")
	}

	reqs.Add(1);
	reqs.Add(3);
	if reqs.i != 4 {
		t.Errorf("reqs.i = %v, want 4", reqs.i)
	}

	if s := reqs.String(); s != "4" {
		t.Errorf("reqs.String() = %q, want \"4\"", s);
	}
}

func TestString(t *testing.T) {
	name := NewString("my-name");
	if name.s != "" {
		t.Errorf("name.s = %q, want \"\"", name.s)
	}

	name.Set("Mike");
	if name.s != "Mike" {
		t.Errorf("name.s = %q, want \"Mike\"", name.s)
	}

	if s := name.String(); s != "\"Mike\"" {
		t.Errorf("reqs.String() = %q, want \"\"Mike\"\"", s);
	}
}

func TestMapCounter(t *testing.T) {
	colours := NewMap("bike-shed-colours");

	colours.Add("red", 1);
	colours.Add("red", 2);
	colours.Add("blue", 4);
	if x := colours.m["red"].(*Int).i; x != 3 {
		t.Errorf("colours.m[\"red\"] = %v, want 3", x)
	}
	if x := colours.m["blue"].(*Int).i; x != 4 {
		t.Errorf("colours.m[\"blue\"] = %v, want 4", x)
	}

	// colours.String() should be '{"red":3, "blue":4}',
	// though the order of red and blue could vary.
	s := colours.String();
	j, ok, errtok := json.StringToJson(s);
	if !ok {
		t.Errorf("colours.String() isn't valid JSON: %v", errtok)
	}
	if j.Kind() != json.MapKind {
		t.Error("colours.String() didn't produce a map.")
	}
	red := j.Get("red");
	if red.Kind() != json.NumberKind {
		t.Error("red.Kind() is not a NumberKind.")
	}
	if x := red.Number(); x != 3 {
		t.Error("red = %v, want 3", x)
	}
}
