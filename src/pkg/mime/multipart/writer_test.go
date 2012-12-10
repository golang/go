// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"io/ioutil"
	"strings"
	"testing"
)

func TestWriter(t *testing.T) {
	fileContents := []byte("my file contents")

	var b bytes.Buffer
	w := NewWriter(&b)
	{
		part, err := w.CreateFormFile("myfile", "my-file.txt")
		if err != nil {
			t.Fatalf("CreateFormFile: %v", err)
		}
		part.Write(fileContents)
		err = w.WriteField("key", "val")
		if err != nil {
			t.Fatalf("WriteField: %v", err)
		}
		part.Write([]byte("val"))
		err = w.Close()
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
		s := b.String()
		if len(s) == 0 {
			t.Fatal("String: unexpected empty result")
		}
		if s[0] == '\r' || s[0] == '\n' {
			t.Fatal("String: unexpected newline")
		}
	}

	r := NewReader(&b, w.Boundary())

	part, err := r.NextPart()
	if err != nil {
		t.Fatalf("part 1: %v", err)
	}
	if g, e := part.FormName(), "myfile"; g != e {
		t.Errorf("part 1: want form name %q, got %q", e, g)
	}
	slurp, err := ioutil.ReadAll(part)
	if err != nil {
		t.Fatalf("part 1: ReadAll: %v", err)
	}
	if e, g := string(fileContents), string(slurp); e != g {
		t.Errorf("part 1: want contents %q, got %q", e, g)
	}

	part, err = r.NextPart()
	if err != nil {
		t.Fatalf("part 2: %v", err)
	}
	if g, e := part.FormName(), "key"; g != e {
		t.Errorf("part 2: want form name %q, got %q", e, g)
	}
	slurp, err = ioutil.ReadAll(part)
	if err != nil {
		t.Fatalf("part 2: ReadAll: %v", err)
	}
	if e, g := "val", string(slurp); e != g {
		t.Errorf("part 2: want contents %q, got %q", e, g)
	}

	part, err = r.NextPart()
	if part != nil || err == nil {
		t.Fatalf("expected end of parts; got %v, %v", part, err)
	}
}

func TestWriterSetBoundary(t *testing.T) {
	var b bytes.Buffer
	w := NewWriter(&b)
	tests := []struct {
		b  string
		ok bool
	}{
		{"abc", true},
		{"", false},
		{"ungültig", false},
		{"!", false},
		{strings.Repeat("x", 69), true},
		{strings.Repeat("x", 70), false},
		{"bad!ascii!", false},
		{"my-separator", true},
	}
	for i, tt := range tests {
		err := w.SetBoundary(tt.b)
		got := err == nil
		if got != tt.ok {
			t.Errorf("%d. boundary %q = %v (%v); want %v", i, tt.b, got, err, tt.ok)
		} else if tt.ok {
			got := w.Boundary()
			if got != tt.b {
				t.Errorf("boundary = %q; want %q", got, tt.b)
			}
		}
	}
	w.Close()
	if got := b.String(); !strings.Contains(got, "\r\n--my-separator--\r\n") {
		t.Errorf("expected my-separator in output. got: %q", got)
	}
}
