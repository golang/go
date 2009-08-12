// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package log

// These tests are too simple.

import (
	"bufio";
	"os";
	"regexp";
	"testing";
)

const (
	Rdate = `[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9]`;
	Rtime = `[0-9][0-9]:[0-9][0-9]:[0-9][0-9]`;
	Rmicroseconds = `\.[0-9][0-9][0-9][0-9][0-9][0-9]`;
	Rline = `[0-9]+:`;
	Rlongfile = `/[A-Za-z0-9_/\-]+\.go:` + Rline;
	Rshortfile = `[A-Za-z0-9_\-]+\.go:` + Rline;
)

type tester struct {
	flag	int;
	prefix	string;
	pattern	string;	// regexp that log output must match; we add ^ and expected_text$ always
}

var tests = []tester {
	// individual pieces:
	tester{ 0,	"", "" },
	tester{ 0, "XXX", "XXX" },
	tester{ Lok|Ldate, "", Rdate+" " },
	tester{ Lok|Ltime, "", Rtime+" " },
	tester{ Lok|Ltime|Lmicroseconds, "", Rtime+Rmicroseconds+" " },
	tester{ Lok|Lmicroseconds, "", Rtime+Rmicroseconds+" " },	// microsec implies time
	tester{ Lok|Llongfile, "", Rlongfile+" " },
	tester{ Lok|Lshortfile, "", Rshortfile+" " },
	tester{ Lok|Llongfile|Lshortfile, "", Rshortfile+" " },	// shortfile overrides longfile
	// everything at once:
	tester{ Lok|Ldate|Ltime|Lmicroseconds|Llongfile, "XXX", "XXX"+Rdate+" "+Rtime+Rmicroseconds+" "+Rlongfile+" " },
	tester{ Lok|Ldate|Ltime|Lmicroseconds|Lshortfile, "XXX", "XXX"+Rdate+" "+Rtime+Rmicroseconds+" "+Rshortfile+" " },
}

// Test using Log("hello", 23, "world") or using Logf("hello %d world", 23)
func testLog(t *testing.T, flag int, prefix string, pattern string, useLogf bool) {
	r, w, err1 := os.Pipe();
	if err1 != nil {
		t.Fatal("pipe", err1);
	}
	defer r.Close();
	defer w.Close();
	buf := bufio.NewReader(r);
	l := New(w, nil, prefix, flag);
	if useLogf {
		l.Logf("hello %d world", 23);
	} else {
		l.Log("hello", 23, "world");
	}
	line, err3 := buf.ReadLineString('\n', false);
	if err3 != nil {
		t.Fatal("log error", err3);
	}
	pattern = "^"+pattern+"hello 23 world$";
	matched, err4 := regexp.MatchString(pattern, line);
	if err4 != nil{
		t.Fatal("pattern did not compile:", err4);
	}
	if !matched {
		t.Errorf("log output should match %q is %q", pattern, line);
	}
}

func TestAllLog(t *testing.T) {
	for i, testcase := range tests {
		testLog(t, testcase.flag, testcase.prefix, testcase.pattern, false);
		testLog(t, testcase.flag, testcase.prefix, testcase.pattern, true);
	}
}
