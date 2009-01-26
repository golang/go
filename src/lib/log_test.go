// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package log

// These tests are too simple.

import (
	"bufio";
	"log";
	"os";
	"testing";
)

func test(t *testing.T, flag int, expect string) {
	fd0, fd1, err1 := os.Pipe();
	if err1 != nil {
		t.Error("pipe", err1);
	}
	buf, err2 := bufio.NewBufRead(fd0);
	if err2 != nil {
		t.Error("bufio.NewBufRead", err2);
	}
	l := NewLogger(fd1, nil, flag);
	l.Log("hello", 23, "world");	/// the line number of this line needs to be placed in the expect strings
	line, err3 := buf.ReadLineString('\n', false);
	if line[len(line)-len(expect):len(line)] != expect {
		t.Error("log output should be ...", expect, "; is " , line);
	}
	t.Log(line);
	fd0.Close();
	fd1.Close();
}

func TestRegularLog(t *testing.T) {
	test(t, Lok, "/go/src/lib/log_test.go:25: hello 23 world");
}

func TestShortNameLog(t *testing.T) {
	test(t, Lok|Lshortname, " log_test.go:25: hello 23 world")
}

func testFormatted(t *testing.T, flag int, expect string) {
	fd0, fd1, err1 := os.Pipe();
	if err1 != nil {
		t.Error("pipe", err1);
	}
	buf, err2 := bufio.NewBufRead(fd0);
	if err2 != nil {
		t.Error("bufio.NewBufRead", err2);
	}
	l := NewLogger(fd1, nil, flag);
	l.Logf("hello %d world", 23);	/// the line number of this line needs to be placed in the expect strings
	line, err3 := buf.ReadLineString('\n', false);
	if line[len(line)-len(expect):len(line)] != expect {
		t.Error("log output should be ...", expect, "; is " , line);
	}
	t.Log(line);
	fd0.Close();
	fd1.Close();
}

func TestRegularLogFormatted(t *testing.T) {
	testFormatted(t, Lok, "/go/src/lib/log_test.go:53: hello 23 world");
}

func TestShortNameLogFormatted(t *testing.T) {
	testFormatted(t, Lok|Lshortname, " log_test.go:53: hello 23 world")
}
