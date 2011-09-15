// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textproto

import (
	"bufio"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
)

type canonicalHeaderKeyTest struct {
	in, out string
}

var canonicalHeaderKeyTests = []canonicalHeaderKeyTest{
	{"a-b-c", "A-B-C"},
	{"a-1-c", "A-1-C"},
	{"User-Agent", "User-Agent"},
	{"uSER-aGENT", "User-Agent"},
	{"user-agent", "User-Agent"},
	{"USER-AGENT", "User-Agent"},
}

func TestCanonicalMIMEHeaderKey(t *testing.T) {
	for _, tt := range canonicalHeaderKeyTests {
		if s := CanonicalMIMEHeaderKey(tt.in); s != tt.out {
			t.Errorf("CanonicalMIMEHeaderKey(%q) = %q, want %q", tt.in, s, tt.out)
		}
	}
}

func reader(s string) *Reader {
	return NewReader(bufio.NewReader(strings.NewReader(s)))
}

func TestReadLine(t *testing.T) {
	r := reader("line1\nline2\n")
	s, err := r.ReadLine()
	if s != "line1" || err != nil {
		t.Fatalf("Line 1: %s, %v", s, err)
	}
	s, err = r.ReadLine()
	if s != "line2" || err != nil {
		t.Fatalf("Line 2: %s, %v", s, err)
	}
	s, err = r.ReadLine()
	if s != "" || err != os.EOF {
		t.Fatalf("EOF: %s, %v", s, err)
	}
}

func TestReadContinuedLine(t *testing.T) {
	r := reader("line1\nline\n 2\nline3\n")
	s, err := r.ReadContinuedLine()
	if s != "line1" || err != nil {
		t.Fatalf("Line 1: %s, %v", s, err)
	}
	s, err = r.ReadContinuedLine()
	if s != "line 2" || err != nil {
		t.Fatalf("Line 2: %s, %v", s, err)
	}
	s, err = r.ReadContinuedLine()
	if s != "line3" || err != nil {
		t.Fatalf("Line 3: %s, %v", s, err)
	}
	s, err = r.ReadContinuedLine()
	if s != "" || err != os.EOF {
		t.Fatalf("EOF: %s, %v", s, err)
	}
}

func TestReadCodeLine(t *testing.T) {
	r := reader("123 hi\n234 bye\n345 no way\n")
	code, msg, err := r.ReadCodeLine(0)
	if code != 123 || msg != "hi" || err != nil {
		t.Fatalf("Line 1: %d, %s, %v", code, msg, err)
	}
	code, msg, err = r.ReadCodeLine(23)
	if code != 234 || msg != "bye" || err != nil {
		t.Fatalf("Line 2: %d, %s, %v", code, msg, err)
	}
	code, msg, err = r.ReadCodeLine(346)
	if code != 345 || msg != "no way" || err == nil {
		t.Fatalf("Line 3: %d, %s, %v", code, msg, err)
	}
	if e, ok := err.(*Error); !ok || e.Code != code || e.Msg != msg {
		t.Fatalf("Line 3: wrong error %v\n", err)
	}
	code, msg, err = r.ReadCodeLine(1)
	if code != 0 || msg != "" || err != os.EOF {
		t.Fatalf("EOF: %d, %s, %v", code, msg, err)
	}
}

func TestReadDotLines(t *testing.T) {
	r := reader("dotlines\r\n.foo\r\n..bar\n...baz\nquux\r\n\r\n.\r\nanother\n")
	s, err := r.ReadDotLines()
	want := []string{"dotlines", "foo", ".bar", "..baz", "quux", ""}
	if !reflect.DeepEqual(s, want) || err != nil {
		t.Fatalf("ReadDotLines: %v, %v", s, err)
	}

	s, err = r.ReadDotLines()
	want = []string{"another"}
	if !reflect.DeepEqual(s, want) || err != io.ErrUnexpectedEOF {
		t.Fatalf("ReadDotLines2: %v, %v", s, err)
	}
}

func TestReadDotBytes(t *testing.T) {
	r := reader("dotlines\r\n.foo\r\n..bar\n...baz\nquux\r\n\r\n.\r\nanot.her\r\n")
	b, err := r.ReadDotBytes()
	want := []byte("dotlines\nfoo\n.bar\n..baz\nquux\n\n")
	if !reflect.DeepEqual(b, want) || err != nil {
		t.Fatalf("ReadDotBytes: %q, %v", b, err)
	}

	b, err = r.ReadDotBytes()
	want = []byte("anot.her\n")
	if !reflect.DeepEqual(b, want) || err != io.ErrUnexpectedEOF {
		t.Fatalf("ReadDotBytes2: %q, %v", b, err)
	}
}

func TestReadMIMEHeader(t *testing.T) {
	r := reader("my-key: Value 1  \r\nLong-key: Even \n Longer Value\r\nmy-Key: Value 2\r\n\n")
	m, err := r.ReadMIMEHeader()
	want := MIMEHeader{
		"My-Key":   {"Value 1", "Value 2"},
		"Long-Key": {"Even Longer Value"},
	}
	if !reflect.DeepEqual(m, want) || err != nil {
		t.Fatalf("ReadMIMEHeader: %v, %v; want %v", m, err, want)
	}
}

type readResponseTest struct {
	in       string
	inCode   int
	wantCode int
	wantMsg  string
}

var readResponseTests = []readResponseTest{
	{"230-Anonymous access granted, restrictions apply\n" +
		"Read the file README.txt,\n" +
		"230  please",
		23,
		230,
		"Anonymous access granted, restrictions apply\nRead the file README.txt,\n please",
	},

	{"230 Anonymous access granted, restrictions apply\n",
		23,
		230,
		"Anonymous access granted, restrictions apply",
	},

	{"400-A\n400-B\n400 C",
		4,
		400,
		"A\nB\nC",
	},

	{"400-A\r\n400-B\r\n400 C\r\n",
		4,
		400,
		"A\nB\nC",
	},
}

// See http://www.ietf.org/rfc/rfc959.txt page 36.
func TestRFC959Lines(t *testing.T) {
	for i, tt := range readResponseTests {
		r := reader(tt.in + "\nFOLLOWING DATA")
		code, msg, err := r.ReadResponse(tt.inCode)
		if err != nil {
			t.Errorf("#%d: ReadResponse: %v", i, err)
			continue
		}
		if code != tt.wantCode {
			t.Errorf("#%d: code=%d, want %d", i, code, tt.wantCode)
		}
		if msg != tt.wantMsg {
			t.Errorf("%#d: msg=%q, want %q", i, msg, tt.wantMsg)
		}
	}
}
