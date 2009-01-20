// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt";
	"flag";
)

var chatty = flag.Bool("chatty", false, "chatty")

// Insert tabs after newlines - but not the last one
func tabify(s string) string {
	for i := 0; i < len(s) - 1; i++ {	// -1 because if last char is newline, don't bother
		if s[i] == '\n' {
			return s[0:i+1] + "\t" + tabify(s[i+1:len(s)]);
		}
	}
	return s
}

type T struct {
	errors	string;
	failed	bool;
	ch	chan *T;
}

func (t *T) Fail() {
	t.failed = true
}

func (t *T) FailNow() {
	t.Fail();
	t.ch <- t;
	sys.Goexit();
}

func (t *T) Log(args ...) {
	t.errors += "\t" + tabify(fmt.Sprintln(args));
}

func (t *T) Logf(format string, args ...) {
	t.errors += tabify(fmt.Sprintf("\t" + format, args));
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

type Test struct {
	Name string;
	F *(*T);
}

func tRunner(t *T, test *Test) {
	test.F(t);
	t.ch <- t;
}

func Main(tests []Test) {
	flag.Parse();
	ok := true;
	if len(tests) == 0 {
		println("testing: warning: no tests to run");
	}
	for i := 0; i < len(tests); i++ {
		if *chatty {
			println("=== RUN ", tests[i].Name);
		}
		t := new(T);
		t.ch = make(chan *T);
		go tRunner(t, &tests[i]);
		<-t.ch;
		if t.failed {
			println("--- FAIL:", tests[i].Name);
			print(t.errors);
			ok = false;
		} else if *chatty {
			println("--- PASS:", tests[i].Name);
			print(t.errors);
		}
	}
	if !ok {
		println("FAIL");
		sys.Exit(1);
	}
	println("PASS");
}
