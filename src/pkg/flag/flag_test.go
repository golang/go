// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag_test

import (
	. "flag"
	"fmt"
	"os"
	"testing"
)

var (
	test_bool    = Bool("test_bool", false, "bool value")
	test_int     = Int("test_int", 0, "int value")
	test_int64   = Int64("test_int64", 0, "int64 value")
	test_uint    = Uint("test_uint", 0, "uint value")
	test_uint64  = Uint64("test_uint64", 0, "uint64 value")
	test_string  = String("test_string", "0", "string value")
	test_float64 = Float64("test_float64", 0, "float64 value")
)

func boolString(s string) string {
	if s == "0" {
		return "false"
	}
	return "true"
}

func TestEverything(t *testing.T) {
	m := make(map[string]*Flag)
	desired := "0"
	visitor := func(f *Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			m[f.Name] = f
			ok := false
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
	}
	VisitAll(visitor)
	if len(m) != 7 {
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
	Set("test_bool", "true")
	Set("test_int", "1")
	Set("test_int64", "1")
	Set("test_uint", "1")
	Set("test_uint64", "1")
	Set("test_string", "1")
	Set("test_float64", "1")
	desired = "1"
	Visit(visitor)
	if len(m) != 7 {
		t.Error("Visit fails after set")
		for k, v := range m {
			t.Log(k, *v)
		}
	}
}

func TestUsage(t *testing.T) {
	called := false
	ResetForTesting(func() { called = true })
	if ParseForTesting([]string{"a.out", "-x"}) {
		t.Error("parse did not fail for unknown flag")
	}
	if !called {
		t.Error("did not call Usage for unknown flag")
	}
}

func TestParse(t *testing.T) {
	ResetForTesting(func() { t.Error("bad parse") })
	boolFlag := Bool("bool", false, "bool value")
	bool2Flag := Bool("bool2", false, "bool2 value")
	intFlag := Int("int", 0, "int value")
	int64Flag := Int64("int64", 0, "int64 value")
	uintFlag := Uint("uint", 0, "uint value")
	uint64Flag := Uint64("uint64", 0, "uint64 value")
	stringFlag := String("string", "0", "string value")
	float64Flag := Float64("float64", 0, "float64 value")
	extra := "one-extra-argument"
	args := []string{
		"a.out",
		"-bool",
		"-bool2=true",
		"--int", "22",
		"--int64", "23",
		"-uint", "24",
		"--uint64", "25",
		"-string", "hello",
		"-float64", "2718e28",
		extra,
	}
	if !ParseForTesting(args) {
		t.Fatal("parse failed")
	}
	if *boolFlag != true {
		t.Error("bool flag should be true, is ", *boolFlag)
	}
	if *bool2Flag != true {
		t.Error("bool2 flag should be true, is ", *bool2Flag)
	}
	if *intFlag != 22 {
		t.Error("int flag should be 22, is ", *intFlag)
	}
	if *int64Flag != 23 {
		t.Error("int64 flag should be 23, is ", *int64Flag)
	}
	if *uintFlag != 24 {
		t.Error("uint flag should be 24, is ", *uintFlag)
	}
	if *uint64Flag != 25 {
		t.Error("uint64 flag should be 25, is ", *uint64Flag)
	}
	if *stringFlag != "hello" {
		t.Error("string flag should be `hello`, is ", *stringFlag)
	}
	if *float64Flag != 2718e28 {
		t.Error("float64 flag should be 2718e28, is ", *float64Flag)
	}
	if len(Args()) != 1 {
		t.Error("expected one argument, got", len(Args()))
	} else if Args()[0] != extra {
		t.Errorf("expected argument %q got %q", extra, Args()[0])
	}
}

// Declare a user-defined flag.
type flagVar []string

func (f *flagVar) String() string {
	return fmt.Sprint([]string(*f))
}

func (f *flagVar) Set(value string) bool {
	*f = append(*f, value)
	return true
}

func TestUserDefined(t *testing.T) {
	ResetForTesting(func() { t.Fatal("bad parse") })
	var v flagVar
	Var(&v, "v", "usage")
	if !ParseForTesting([]string{"a.out", "-v", "1", "-v", "2", "-v=3"}) {
		t.Error("parse failed")
	}
	if len(v) != 3 {
		t.Fatal("expected 3 args; got ", len(v))
	}
	expect := "[1 2 3]"
	if v.String() != expect {
		t.Errorf("expected value %q got %q", expect, v.String())
	}
}

func TestChangingArgs(t *testing.T) {
	ResetForTesting(func() { t.Fatal("bad parse") })
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = []string{"cmd", "-before", "subcmd", "-after", "args"}
	before := Bool("before", false, "")
	Parse()
	cmd := Arg(0)
	os.Args = Args()
	after := Bool("after", false, "")
	Parse()
	args := Args()

	if !*before || cmd != "subcmd" || !*after || len(args) != 1 || args[0] != "args" {
		t.Fatalf("expected true subcmd true [args] got %v %v %v %v", *before, cmd, *after, args)
	}
}
