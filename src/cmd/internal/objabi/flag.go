// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"flag"
	"fmt"
	"os"
	"strconv"
)

func Flagcount(name, usage string, val *int) {
	flag.Var((*count)(val), name, usage)
}

func Flagfn0(name, usage string, f func()) {
	flag.Var(fn0(f), name, usage)
}

func Flagfn1(name, usage string, f func(string)) {
	flag.Var(fn1(f), name, usage)
}

func Flagprint(fd int) {
	if fd == 1 {
		flag.CommandLine.SetOutput(os.Stdout)
	}
	flag.PrintDefaults()
}

func Flagparse(usage func()) {
	flag.Usage = usage
	flag.Parse()
}

// count is a flag.Value that is like a flag.Bool and a flag.Int.
// If used as -name, it increments the count, but -name=x sets the count.
// Used for verbose flag -v.
type count int

func (c *count) String() string {
	return fmt.Sprint(int(*c))
}

func (c *count) Set(s string) error {
	switch s {
	case "true":
		*c++
	case "false":
		*c = 0
	default:
		n, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("invalid count %q", s)
		}
		*c = count(n)
	}
	return nil
}

func (c *count) IsBoolFlag() bool {
	return true
}

type fn0 func()

func (f fn0) Set(s string) error {
	f()
	return nil
}

func (f fn0) Get() interface{} { return nil }

func (f fn0) String() string { return "" }

func (f fn0) IsBoolFlag() bool {
	return true
}

type fn1 func(string)

func (f fn1) Set(s string) error {
	f(s)
	return nil
}

func (f fn1) String() string { return "" }
