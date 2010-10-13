// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package log

// These tests are too simple.

import (
	"bytes"
	"os"
	"regexp"
	"testing"
)

const (
	Rdate         = `[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]`
	Rtime         = `[0-9][0-9]:[0-9][0-9]:[0-9][0-9]`
	Rmicroseconds = `\.[0-9][0-9][0-9][0-9][0-9][0-9]`
	Rline         = `(54|56):` // must update if the calls to l.Printf / l.Print below move
	Rlongfile     = `.*/[A-Za-z0-9_\-]+\.go:` + Rline
	Rshortfile    = `[A-Za-z0-9_\-]+\.go:` + Rline
)

type tester struct {
	flag    int
	prefix  string
	pattern string // regexp that log output must match; we add ^ and expected_text$ always
}

var tests = []tester{
	// individual pieces:
	tester{0, "", ""},
	tester{0, "XXX", "XXX"},
	tester{Ldate, "", Rdate + " "},
	tester{Ltime, "", Rtime + " "},
	tester{Ltime | Lmicroseconds, "", Rtime + Rmicroseconds + " "},
	tester{Lmicroseconds, "", Rtime + Rmicroseconds + " "}, // microsec implies time
	tester{Llongfile, "", Rlongfile + " "},
	tester{Lshortfile, "", Rshortfile + " "},
	tester{Llongfile | Lshortfile, "", Rshortfile + " "}, // shortfile overrides longfile
	// everything at once:
	tester{Ldate | Ltime | Lmicroseconds | Llongfile, "XXX", "XXX" + Rdate + " " + Rtime + Rmicroseconds + " " + Rlongfile + " "},
	tester{Ldate | Ltime | Lmicroseconds | Lshortfile, "XXX", "XXX" + Rdate + " " + Rtime + Rmicroseconds + " " + Rshortfile + " "},
}

// Test using Println("hello", 23, "world") or using Printf("hello %d world", 23)
func testPrint(t *testing.T, flag int, prefix string, pattern string, useFormat bool) {
	buf := new(bytes.Buffer)
	SetOutput(buf)
	SetFlags(flag)
	SetPrefix(prefix)
	if useFormat {
		Printf("hello %d world", 23)
	} else {
		Println("hello", 23, "world")
	}
	line := buf.String()
	line = line[0 : len(line)-1]
	pattern = "^" + pattern + "hello 23 world$"
	matched, err4 := regexp.MatchString(pattern, line)
	if err4 != nil {
		t.Fatal("pattern did not compile:", err4)
	}
	if !matched {
		t.Errorf("log output should match %q is %q", pattern, line)
	}
	SetOutput(os.Stderr)
}

func TestAll(t *testing.T) {
	for _, testcase := range tests {
		testPrint(t, testcase.flag, testcase.prefix, testcase.pattern, false)
		testPrint(t, testcase.flag, testcase.prefix, testcase.pattern, true)
	}
}
