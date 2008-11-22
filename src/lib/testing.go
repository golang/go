// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt";
	"flag";
)

var chatty bool;
func init() {
	flag.Bool("chatty", false, &chatty, "chatty");
}

// Insert tabs after newlines - but not the last one
func Tabify(s string) string {
	for i := 0; i < len(s) - 1; i++ {	// -1 because if last char is newline, don't bother
		if s[i] == '\n' {
			return s[0:i+1] + "\t" + Tabify(s[i+1:len(s)]);
		}
	}
	return s
}

export type T struct {
	errors	string;
	failed	bool;
	ch	*chan *T;
}

func (t *T) Fail() {
	t.failed = true
}

func (t *T) FailNow() {
	t.Fail();
	t.ch <- t;
	sys.goexit();
}

func (t *T) Log(args ...) {
	t.errors += "\t" + Tabify(fmt.sprintln(args));
}

func (t *T) Logf(format string, args ...) {
	t.errors += Tabify(fmt.sprintf("\t" + format, args));
	l := len(t.errors);
	if l > 0 && t.errors[l-1] != '\n' {
		t.errors += "\n"
	}
}

func (t *T) Error(args ...) {
	t.Log(args);
	t.Fail();
}

func (t *T) Errorf(format string, args ...) {
	t.Logf(format, args);
	t.Fail();
}

func (t *T) Fatal(args ...) {
	t.Log(args);
	t.FailNow();
}

func (t *T) Fatalf(format string, args ...) {
	t.Logf(format, args);
	t.FailNow();
}

export type Test struct {
	name string;
	f *(*T);
}

func TRunner(t *T, test *Test) {
	test.f(t);
	t.ch <- t;
}

export func Main(tests *[]Test) {
	flag.Parse();
	ok := true;
	if len(tests) == 0 {
		println("gotest: warning: no tests to run");
	}
	for i := 0; i < len(tests); i++ {
		if chatty {
			println("=== RUN ", tests[i].name);
		}
		t := new(T);
		t.ch = new(chan *T);
		go TRunner(t, &tests[i]);
		<-t.ch;
		if t.failed {
			println("--- FAIL:", tests[i].name);
			print(t.errors);
			ok = false;
		} else if chatty {
			println("--- PASS:", tests[i].name);
			print(t.errors);
		}
	}
	if !ok {
		println("FAIL");
		sys.exit(1);
	}
	println("PASS");
}
