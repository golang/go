// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag

import (
	"flag";
	"fmt";
	"testing";
)

var (
	test_bool = flag.Bool("test_bool", false, "bool value");
	test_int = flag.Int("test_int", 0, "int value");
	test_int64 = flag.Int64("test_int64", 0, "int64 value");
	test_uint = flag.Uint("test_uint", 0, "uint value");
	test_uint64 = flag.Uint64("test_uint64", 0, "uint64 value");
	test_string = flag.String("test_string", "0", "string value");
)

func boolString(s string) string {
	if s == "0" {
		return "false"
	}
	return "true"
}

func TestEverything(t *testing.T) {
	m := make(map[string] *flag.Flag);
	desired := "0";
	visitor := func(f *flag.Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			m[f.Name] = f;
			ok := false;
			switch {
			case f.Value.String() == desired:
				ok = true;
			case f.Name == "test_bool" && f.Value.String() == boolString(desired):
				ok = true;
			}
			if !ok {
				t.Error("flag.Visit: bad value", f.Value.String(), "for", f.Name);
			}
		}
	};
	flag.VisitAll(visitor);
	if len(m) != 6 {
		t.Error("flag.VisitAll misses some flags");
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	m = make(map[string] *flag.Flag);
	flag.Visit(visitor);
	if len(m) != 0 {
		t.Errorf("flag.Visit sees unset flags");
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	// Now set all flags
	flag.Set("test_bool", "true");
	flag.Set("test_int", "1");
	flag.Set("test_int64", "1");
	flag.Set("test_uint", "1");
	flag.Set("test_uint64", "1");
	flag.Set("test_string", "1");
	desired = "1";
	flag.Visit(visitor);
	if len(m) != 6 {
		t.Error("flag.Visit fails after set");
		for k, v := range m {
			t.Log(k, *v)
		}
	}
}
