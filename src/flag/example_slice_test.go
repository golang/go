// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag_test

import (
	. "flag"
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"
)

func TestSliceEverything(t *testing.T) {
	ResetForTesting(nil)

	var (
		bools     = []bool{false, true}
		ints      = []int{1, 2}
		int64s    = []int64{1, 2}
		uints     = []uint{1, 2}
		uint64s   = []uint64{1, 2}
		strs      = []string{"go", "hello"}
		float64s  = []float64{1, 2}
		durations = []time.Duration{0, 0}
	)
	flagLen := 8

	Bools("test_bools", bools, "bools value")
	Ints("test_ints", ints, "ints value")
	Int64s("test_int64s", int64s, "int64s value")
	Uints("test_uints", uints, "uints value")
	Uint64s("test_uint64s", uint64s, "uint64 value")
	Strings("test_strings", strs, "strings value")
	Float64s("test_float64s", float64s, "float64s value")
	Durations("test_durations", durations, "time.Durations value")

	m := make(map[string]*Flag)
	visitor := func(f *Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			m[f.Name] = f
			ok := false
			switch {
			case f.Name == "test_bools" && f.Value.String() == fmt.Sprint(bools):
				ok = true
			case f.Name == "test_ints" && f.Value.String() == fmt.Sprint(ints):
				ok = true
			case f.Name == "test_int64s" && f.Value.String() == fmt.Sprint(int64s):
				ok = true
			case f.Name == "test_uints" && f.Value.String() == fmt.Sprint(uints):
				ok = true
			case f.Name == "test_uint64s" && f.Value.String() == fmt.Sprint(uint64s):
				ok = true
			case f.Name == "test_strings" && f.Value.String() == fmt.Sprint(strs):
				ok = true
			case f.Name == "test_float64s" && f.Value.String() == fmt.Sprint(float64s):
				ok = true
			case f.Name == "test_durations" && f.Value.String() == fmt.Sprint(durations):
				ok = true
			}
			if !ok {
				t.Error("Visit: bad value", f.Value.String(), "for", f.Name)
			}
		}
	}
	VisitAll(visitor)
	if len(m) != flagLen {
		t.Error("VisitAll misses some flags")
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	m = make(map[string]*Flag)
	Visit(visitor)
	if len(m) != 0 {
		t.Errorf("Visit sees unset flags")
		for k, v := range m {
			t.Log(k, *v)
		}
	}

	// Now set all flags
	Set("test_bools", "false")
	Set("test_bools", "true")

	Set("test_ints", "1")
	Set("test_ints", "2")

	Set("test_int64s", "1")
	Set("test_int64s", "2")

	Set("test_uints", "1")
	Set("test_uints", "2")

	Set("test_uint64s", "1")
	Set("test_uint64s", "2")

	Set("test_strings", "go")
	Set("test_strings", "hello")

	Set("test_float64s", "1")
	Set("test_float64s", "2")

	Set("test_durations", "0")
	Set("test_durations", "0")

	visitorAppend := func(f *Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			m[f.Name] = f
			ok := false
			switch {
			case f.Name == "test_bools":
				bools = append(bools, false, true)
				ok = f.Value.String() == fmt.Sprint(bools)
			case f.Name == "test_ints":
				ints = append(ints, 1, 2)
				ok = f.Value.String() == fmt.Sprint(ints)
			case f.Name == "test_int64s":
				int64s = append(int64s, 1, 2)
				ok = f.Value.String() == fmt.Sprint(int64s)
			case f.Name == "test_uints":
				uints = append(uints, 1, 2)
				ok = f.Value.String() == fmt.Sprint(uints)
			case f.Name == "test_uint64s":
				uint64s = append(uint64s, 1, 2)
				ok = f.Value.String() == fmt.Sprint(uint64s)
			case f.Name == "test_strings":
				strs = append(strs, "go", "hello")
				ok = f.Value.String() == fmt.Sprint(strs)
			case f.Name == "test_float64s":
				float64s = append(float64s, 1, 2)
				ok = f.Value.String() == fmt.Sprint(float64s)
			case f.Name == "test_durations":
				durations = append(durations, 0, 0)
				ok = f.Value.String() == fmt.Sprint(durations)
			}
			if !ok {
				t.Error("Visit Append: bad value", f.Value.String(), "for", f.Name)
			}
		}
	}

	Visit(visitorAppend)

	if len(m) != flagLen {
		t.Error("Visit fails after set")
		for k, v := range m {
			t.Log(k, *v)
		}
	}

	// Now test they're visited in sort order.
	var flagNames []string
	Visit(func(f *Flag) { flagNames = append(flagNames, f.Name) })
	if !sort.StringsAreSorted(flagNames) {
		t.Errorf("flag names not sorted: %v", flagNames)
	}
}

func TestSliceGet(t *testing.T) {
	ResetForTesting(nil)

	var (
		bools     = []bool{false, true}
		ints      = []int{1, 2}
		int64s    = []int64{1, 2}
		uints     = []uint{1, 2}
		uint64s   = []uint64{1, 2}
		strings   = []string{"go", "hello"}
		float64s  = []float64{1, 2}
		durations = []time.Duration{0, 0}
	)

	Bools("test_bools", bools, "bools value")
	Ints("test_ints", ints, "ints value")
	Int64s("test_int64s", int64s, "int64s value")
	Uints("test_uints", uints, "uints value")
	Uint64s("test_uint64s", uint64s, "uint64 value")
	Strings("test_strings", strings, "strings value")
	Float64s("test_float64s", float64s, "float64s value")
	Durations("test_durations", durations, "time.Durations value")

	visitor := func(f *Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			g, ok := f.Value.(Getter)
			if !ok {
				t.Errorf("Visit: value does not satisfy Getter: %T", f.Value)
				return
			}
			switch f.Name {
			case "test_bools":
				ok = reflect.DeepEqual(g.Get(), bools)
			case "test_ints":
				ok = reflect.DeepEqual(g.Get(), ints)
			case "test_int64s":
				ok = reflect.DeepEqual(g.Get(), int64s)
			case "test_uints":
				ok = reflect.DeepEqual(g.Get(), uints)
			case "test_uint64s":
				ok = reflect.DeepEqual(g.Get(), uint64s)
			case "test_strings":
				ok = reflect.DeepEqual(g.Get(), strings)
			case "test_float64s":
				ok = reflect.DeepEqual(g.Get(), float64s)
			case "test_durations":
				ok = reflect.DeepEqual(g.Get(), durations)
			}
			if !ok {
				t.Errorf("Visit: bad value %T(%v) for %s", g.Get(), g.Get(), f.Name)
			}
		}
	}
	VisitAll(visitor)
}

func TestSliceParse(t *testing.T) {
	ResetForTesting(func() { t.Error("bad parse") })
	testSliceParse(CommandLine, t)
}

func testSliceParse(f *FlagSet, t *testing.T) {
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}

	boolsFlag := f.Bools("bools", []bool{false}, "bools value")
	intsFlag := f.Ints("ints", []int{1}, "ints value")
	int64sFlag := f.Int64s("int64s", []int64{1}, "int64s value")
	uintsFlag := f.Uints("uints", []uint{1}, "uints value")
	uint64sFlag := f.Uint64s("uint64s", []uint64{1}, "uint64s value")
	stringsFlag := f.Strings("strings", []string{"go"}, "strings value")
	float64sFlag := f.Float64s("float64s", []float64{0}, "float64s value")
	durationsFlag := f.Durations("durations", []time.Duration{5 * time.Second}, "time.Durations value")

	extra := "one-extra-argument"
	args := []string{
		"-bools",
		"-bools=true",
		"--ints", "22",
		"--int64s", "0x23",
		"-uints", "24",
		"--uint64s", "25",
		"-strings", "hello",
		"-float64s", "2718e28",
		"-durations", "2s",
		extra,
	}
	if err := f.Parse(args); err != nil {
		t.Fatal(err)
	}
	if !f.Parsed() {
		t.Error("f.Parse() = false after Parse")
	}
	if !reflect.DeepEqual(*boolsFlag, []bool{false, true, true}) {
		t.Error("bools flag should be true, is [false, true, true]", *boolsFlag)
	}
	if !reflect.DeepEqual(*intsFlag, []int{1, 22}) {
		t.Error("ints flag should be [1,22], is ", *intsFlag)
	}
	if !reflect.DeepEqual(*int64sFlag, []int64{1, 0x23}) {
		t.Error("int64s flag should be [1,0x23], is ", *int64sFlag)
	}
	if !reflect.DeepEqual(*uintsFlag, []uint{1, 24}) {
		t.Error("uints flag should be [1,24], is ", *uintsFlag)
	}
	if !reflect.DeepEqual(*uint64sFlag, []uint64{1, 25}) {
		t.Error("uint64s flag should be [1,25], is ", *uint64sFlag)
	}
	if !reflect.DeepEqual(*stringsFlag, []string{"go", "hello"}) {
		t.Error(`strings flag should be ["go","hello"], is `, *stringsFlag)
	}
	if !reflect.DeepEqual(*float64sFlag, []float64{0, 2718e28}) {
		t.Error("float64s flag should be [0,2718e28], is ", *float64sFlag)
	}
	if !reflect.DeepEqual(*durationsFlag, []time.Duration{5 * time.Second, 2 * time.Second}) {
		t.Error("durations flag should be [5s,2s], is ", *durationsFlag)
	}

	if len(f.Args()) != 1 {
		t.Error("expected one argument, got", len(f.Args()))
	} else if f.Args()[0] != extra {
		t.Errorf("expected argument %q got %q", extra, f.Args()[0])
	}
}
