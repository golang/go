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
	test_bool = flag.Bool("test_bool", true, "bool value");
	test_int = flag.Int("test_int", 1, "int value");
	test_int64 = flag.Int64("test_int64", 1, "int64 value");
	test_uint = flag.Uint("test_uint", 1, "uint value");
	test_uint64 = flag.Uint64("test_uint64", 1, "uint64 value");
	test_string = flag.String("test_string", "1", "string value");
)

// Because this calls flag.Parse, it needs to be the only Test* function
func TestEverything(t *testing.T) {
	flag.Parse();
	m := make(map[string] *flag.Flag);
	visitor := func(f *flag.Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			m[f.Name] = f
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
	// Now set some flags
	flag.Set("test_bool", "false");
	flag.Set("test_uint", "1234");
	flag.Visit(visitor);
	if len(m) != 2 {
		t.Error("flag.Visit fails after set");
		for k, v := range m {
			t.Log(k, *v)
		}
	}
}
