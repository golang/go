// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag_test

import (
	. "flag";
	"testing";
)

var (
	test_bool	= Bool("test_bool", false, "bool value");
	test_int	= Int("test_int", 0, "int value");
	test_int64	= Int64("test_int64", 0, "int64 value");
	test_uint	= Uint("test_uint", 0, "uint value");
	test_uint64	= Uint64("test_uint64", 0, "uint64 value");
	test_string	= String("test_string", "0", "string value");
	test_float	= Float("test_float", 0, "float value");
	test_float64	= Float("test_float64", 0, "float64 value");
)

func boolString(s string) string {
	if s == "0" {
		return "false"
	}
	return "true";
}

func TestEverything(t *testing.T) {
	m := make(map[string]*Flag);
	desired := "0";
	visitor := func(f *Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			m[f.Name] = f;
			ok := false;
			switch {
			case f.Value.String() == desired:
				ok = true
			case f.Name == "test_bool" && f.Value.String() == boolString(desired):
				ok = true
			}
			if !ok {
				t.Error("Visit: bad value", f.Value.String(), "for", f.Name)
			}
		}
	};
	VisitAll(visitor);
	if len(m) != 8 {
		t.Error("VisitAll misses some flags");
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	m = make(map[string]*Flag);
	Visit(visitor);
	if len(m) != 0 {
		t.Errorf("Visit sees unset flags");
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	// Now set all flags
	Set("test_bool", "true");
	Set("test_int", "1");
	Set("test_int64", "1");
	Set("test_uint", "1");
	Set("test_uint64", "1");
	Set("test_string", "1");
	Set("test_float", "1");
	Set("test_float64", "1");
	desired = "1";
	Visit(visitor);
	if len(m) != 8 {
		t.Error("Visit fails after set");
		for k, v := range m {
			t.Log(k, *v)
		}
	}
}
