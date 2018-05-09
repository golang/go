// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux,!appengine netbsd openbsd windows plan9 solaris

package terminal

import (
	"bytes"
	"io"
	"os"
	"testing"
)

type MockTerminal struct {
	toSend       []byte
	bytesPerRead int
	received     []byte
}

func (c *MockTerminal) Read(data []byte) (n int, err error) {
	n = len(data)
	if n == 0 {
		return
	}
	if n > len(c.toSend) {
		n = len(c.toSend)
	}
	if n == 0 {
		return 0, io.EOF
	}
	if c.bytesPerRead > 0 && n > c.bytesPerRead {
		n = c.bytesPerRead
	}
	copy(data, c.toSend[:n])
	c.toSend = c.toSend[n:]
	return
}

func (c *MockTerminal) Write(data []byte) (n int, err error) {
	c.received = append(c.received, data...)
	return len(data), nil
}

func TestClose(t *testing.T) {
	c := &MockTerminal{}
	ss := NewTerminal(c, "> ")
	line, err := ss.ReadLine()
	if line != "" {
		t.Errorf("Expected empty line but got: %s", line)
	}
	if err != io.EOF {
		t.Errorf("Error should have been EOF but got: %s", err)
	}
}

var keyPressTests = []struct {
	in             string
	line           string
	err            error
	throwAwayLines int
}{
	{
		err: io.EOF,
	},
	{
		in:   "\r",
		line: "",
	},
	{
		in:   "foo\r",
		line: "foo",
	},
	{
		in:   "a\x1b[Cb\r", // right
		line: "ab",
	},
	{
		in:   "a\x1b[Db\r", // left
		line: "ba",
	},
	{
		in:   "a\177b\r", // backspace
		line: "b",
	},
	{
		in: "\x1b[A\r", // up
	},
	{
		in: "\x1b[B\r", // down
	},
	{
		in:   "line\x1b[A\x1b[B\r", // up then down
		line: "line",
	},
	{
		in:             "line1\rline2\x1b[A\r", // recall previous line.
		line:           "line1",
		throwAwayLines: 1,
	},
	{
		// recall two previous lines and append.
		in:             "line1\rline2\rline3\x1b[A\x1b[Axxx\r",
		line:           "line1xxx",
		throwAwayLines: 2,
	},
	{
		// Ctrl-A to move to beginning of line followed by ^K to kill
		// line.
		in:   "a b \001\013\r",
		line: "",
	},
	{
		// Ctrl-A to move to beginning of line, Ctrl-E to move to end,
		// finally ^K to kill nothing.
		in:   "a b \001\005\013\r",
		line: "a b ",
	},
	{
		in:   "\027\r",
		line: "",
	},
	{
		in:   "a\027\r",
		line: "",
	},
	{
		in:   "a \027\r",
		line: "",
	},
	{
		in:   "a b\027\r",
		line: "a ",
	},
	{
		in:   "a b \027\r",
		line: "a ",
	},
	{
		in:   "one two thr\x1b[D\027\r",
		line: "one two r",
	},
	{
		in:   "\013\r",
		line: "",
	},
	{
		in:   "a\013\r",
		line: "a",
	},
	{
		in:   "ab\x1b[D\013\r",
		line: "a",
	},
	{
		in:   "Ξεσκεπάζω\r",
		line: "Ξεσκεπάζω",
	},
	{
		in:             "£\r\x1b[A\177\r", // non-ASCII char, enter, up, backspace.
		line:           "",
		throwAwayLines: 1,
	},
	{
		in:             "£\r££\x1b[A\x1b[B\177\r", // non-ASCII char, enter, 2x non-ASCII, up, down, backspace, enter.
		line:           "£",
		throwAwayLines: 1,
	},
	{
		// Ctrl-D at the end of the line should be ignored.
		in:   "a\004\r",
		line: "a",
	},
	{
		// a, b, left, Ctrl-D should erase the b.
		in:   "ab\x1b[D\004\r",
		line: "a",
	},
	{
		// a, b, c, d, left, left, ^U should erase to the beginning of
		// the line.
		in:   "abcd\x1b[D\x1b[D\025\r",
		line: "cd",
	},
	{
		// Bracketed paste mode: control sequences should be returned
		// verbatim in paste mode.
		in:   "abc\x1b[200~de\177f\x1b[201~\177\r",
		line: "abcde\177",
	},
	{
		// Enter in bracketed paste mode should still work.
		in:             "abc\x1b[200~d\refg\x1b[201~h\r",
		line:           "efgh",
		throwAwayLines: 1,
	},
	{
		// Lines consisting entirely of pasted data should be indicated as such.
		in:   "\x1b[200~a\r",
		line: "a",
		err:  ErrPasteIndicator,
	},
}

func TestKeyPresses(t *testing.T) {
	for i, test := range keyPressTests {
		for j := 1; j < len(test.in); j++ {
			c := &MockTerminal{
				toSend:       []byte(test.in),
				bytesPerRead: j,
			}
			ss := NewTerminal(c, "> ")
			for k := 0; k < test.throwAwayLines; k++ {
				_, err := ss.ReadLine()
				if err != nil {
					t.Errorf("Throwaway line %d from test %d resulted in error: %s", k, i, err)
				}
			}
			line, err := ss.ReadLine()
			if line != test.line {
				t.Errorf("Line resulting from test %d (%d bytes per read) was '%s', expected '%s'", i, j, line, test.line)
				break
			}
			if err != test.err {
				t.Errorf("Error resulting from test %d (%d bytes per read) was '%v', expected '%v'", i, j, err, test.err)
				break
			}
		}
	}
}

func TestPasswordNotSaved(t *testing.T) {
	c := &MockTerminal{
		toSend:       []byte("password\r\x1b[A\r"),
		bytesPerRead: 1,
	}
	ss := NewTerminal(c, "> ")
	pw, _ := ss.ReadPassword("> ")
	if pw != "password" {
		t.Fatalf("failed to read password, got %s", pw)
	}
	line, _ := ss.ReadLine()
	if len(line) > 0 {
		t.Fatalf("password was saved in history")
	}
}

var setSizeTests = []struct {
	width, height int
}{
	{40, 13},
	{80, 24},
	{132, 43},
}

func TestTerminalSetSize(t *testing.T) {
	for _, setSize := range setSizeTests {
		c := &MockTerminal{
			toSend:       []byte("password\r\x1b[A\r"),
			bytesPerRead: 1,
		}
		ss := NewTerminal(c, "> ")
		ss.SetSize(setSize.width, setSize.height)
		pw, _ := ss.ReadPassword("Password: ")
		if pw != "password" {
			t.Fatalf("failed to read password, got %s", pw)
		}
		if string(c.received) != "Password: \r\n" {
			t.Errorf("failed to set the temporary prompt expected %q, got %q", "Password: ", c.received)
		}
	}
}

func TestReadPasswordLineEnd(t *testing.T) {
	var tests = []struct {
		input string
		want  string
	}{
		{"\n", ""},
		{"\r\n", ""},
		{"test\r\n", "test"},
		{"testtesttesttes\n", "testtesttesttes"},
		{"testtesttesttes\r\n", "testtesttesttes"},
		{"testtesttesttesttest\n", "testtesttesttesttest"},
		{"testtesttesttesttest\r\n", "testtesttesttesttest"},
	}
	for _, test := range tests {
		buf := new(bytes.Buffer)
		if _, err := buf.WriteString(test.input); err != nil {
			t.Fatal(err)
		}

		have, err := readPasswordLine(buf)
		if err != nil {
			t.Errorf("readPasswordLine(%q) failed: %v", test.input, err)
			continue
		}
		if string(have) != test.want {
			t.Errorf("readPasswordLine(%q) returns %q, but %q is expected", test.input, string(have), test.want)
			continue
		}

		if _, err = buf.WriteString(test.input); err != nil {
			t.Fatal(err)
		}
		have, err = readPasswordLine(buf)
		if err != nil {
			t.Errorf("readPasswordLine(%q) failed: %v", test.input, err)
			continue
		}
		if string(have) != test.want {
			t.Errorf("readPasswordLine(%q) returns %q, but %q is expected", test.input, string(have), test.want)
			continue
		}
	}
}

func TestMakeRawState(t *testing.T) {
	fd := int(os.Stdout.Fd())
	if !IsTerminal(fd) {
		t.Skip("stdout is not a terminal; skipping test")
	}

	st, err := GetState(fd)
	if err != nil {
		t.Fatalf("failed to get terminal state from GetState: %s", err)
	}
	defer Restore(fd, st)
	raw, err := MakeRaw(fd)
	if err != nil {
		t.Fatalf("failed to get terminal state from MakeRaw: %s", err)
	}

	if *st != *raw {
		t.Errorf("states do not match; was %v, expected %v", raw, st)
	}
}

func TestOutputNewlines(t *testing.T) {
	// \n should be changed to \r\n in terminal output.
	buf := new(bytes.Buffer)
	term := NewTerminal(buf, ">")

	term.Write([]byte("1\n2\n"))
	output := string(buf.Bytes())
	const expected = "1\r\n2\r\n"

	if output != expected {
		t.Errorf("incorrect output: was %q, expected %q", output, expected)
	}
}
