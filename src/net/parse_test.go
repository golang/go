// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bufio"
	"os"
	"runtime"
	"testing"
)

func TestReadLine(t *testing.T) {
	// /etc/services file does not exist on android, plan9, windows.
	switch runtime.GOOS {
	case "android", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	filename := "/etc/services" // a nice big file

	fd, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer fd.Close()
	br := bufio.NewReader(fd)

	file, err := open(filename)
	if file == nil {
		t.Fatal(err)
	}
	defer file.close()

	lineno := 1
	byteno := 0
	for {
		bline, berr := br.ReadString('\n')
		if n := len(bline); n > 0 {
			bline = bline[0 : n-1]
		}
		line, ok := file.readLine()
		if (berr != nil) != !ok || bline != line {
			t.Fatalf("%s:%d (#%d)\nbufio => %q, %v\nnet => %q, %v", filename, lineno, byteno, bline, berr, line, ok)
		}
		if !ok {
			break
		}
		lineno++
		byteno += len(line) + 1
	}
}

func TestGoDebugString(t *testing.T) {
	defer os.Setenv("GODEBUG", os.Getenv("GODEBUG"))
	tests := []struct {
		godebug string
		key     string
		want    string
	}{
		{"", "foo", ""},
		{"foo=", "foo", ""},
		{"foo=bar", "foo", "bar"},
		{"foo=bar,", "foo", "bar"},
		{"foo,foo=bar,", "foo", "bar"},
		{"foo1=bar,foo=bar,", "foo", "bar"},
		{"foo=bar,foo=bar,", "foo", "bar"},
		{"foo=", "foo", ""},
		{"foo", "foo", ""},
		{",foo", "foo", ""},
		{"foo=bar,baz", "loooooooong", ""},
	}
	for _, tt := range tests {
		os.Setenv("GODEBUG", tt.godebug)
		if got := goDebugString(tt.key); got != tt.want {
			t.Errorf("for %q, goDebugString(%q) = %q; want %q", tt.godebug, tt.key, got, tt.want)
		}
	}
}

func TestDtoi(t *testing.T) {
	for _, tt := range []struct {
		in  string
		out int
		off int
		ok  bool
	}{
		{"", 0, 0, false},
		{"0", 0, 1, true},
		{"65536", 65536, 5, true},
		{"123456789", big, 8, false},
		{"-0", 0, 0, false},
		{"-1234", 0, 0, false},
	} {
		n, i, ok := dtoi(tt.in)
		if n != tt.out || i != tt.off || ok != tt.ok {
			t.Errorf("got %d, %d, %v; want %d, %d, %v", n, i, ok, tt.out, tt.off, tt.ok)
		}
	}
}
