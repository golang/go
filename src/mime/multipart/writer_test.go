// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"io"
	"mime"
	"net/textproto"
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
	slurp, err := io.ReadAll(part)
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
	slurp, err = io.ReadAll(part)
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
	tests := []struct {
		b  string
		ok bool
	}{
		{"abc", true},
		{"", false},
		{"ungültig", false},
		{"!", false},
		{strings.Repeat("x", 70), true},
		{strings.Repeat("x", 71), false},
		{"bad!ascii!", false},
		{"my-separator", true},
		{"with space", true},
		{"badspace ", false},
		{"(boundary)", true},
	}
	for i, tt := range tests {
		var b strings.Builder
		w := NewWriter(&b)
		err := w.SetBoundary(tt.b)
		got := err == nil
		if got != tt.ok {
			t.Errorf("%d. boundary %q = %v (%v); want %v", i, tt.b, got, err, tt.ok)
		} else if tt.ok {
			got := w.Boundary()
			if got != tt.b {
				t.Errorf("boundary = %q; want %q", got, tt.b)
			}

			ct := w.FormDataContentType()
			mt, params, err := mime.ParseMediaType(ct)
			if err != nil {
				t.Errorf("could not parse Content-Type %q: %v", ct, err)
			} else if mt != "multipart/form-data" {
				t.Errorf("unexpected media type %q; want %q", mt, "multipart/form-data")
			} else if b := params["boundary"]; b != tt.b {
				t.Errorf("unexpected boundary parameter %q; want %q", b, tt.b)
			}

			w.Close()
			wantSub := "\r\n--" + tt.b + "--\r\n"
			if got := b.String(); !strings.Contains(got, wantSub) {
				t.Errorf("expected %q in output. got: %q", wantSub, got)
			}
		}
	}
}

func TestWriterBoundaryGoroutines(t *testing.T) {
	// Verify there's no data race accessing any lazy boundary if it's used by
	// different goroutines. This was previously broken by
	// https://codereview.appspot.com/95760043/ and reverted in
	// https://codereview.appspot.com/117600043/
	w := NewWriter(io.Discard)
	done := make(chan int)
	go func() {
		w.CreateFormField("foo")
		done <- 1
	}()
	w.Boundary()
	<-done
}

func TestSortedHeader(t *testing.T) {
	var buf strings.Builder
	w := NewWriter(&buf)
	if err := w.SetBoundary("MIMEBOUNDARY"); err != nil {
		t.Fatalf("Error setting mime boundary: %v", err)
	}

	header := textproto.MIMEHeader{
		"A": {"2"},
		"B": {"5", "7", "6"},
		"C": {"4"},
		"M": {"3"},
		"Z": {"1"},
	}

	part, err := w.CreatePart(header)
	if err != nil {
		t.Fatalf("Unable to create part: %v", err)
	}
	part.Write([]byte("foo"))

	w.Close()

	want := "--MIMEBOUNDARY\r\nA: 2\r\nB: 5\r\nB: 7\r\nB: 6\r\nC: 4\r\nM: 3\r\nZ: 1\r\n\r\nfoo\r\n--MIMEBOUNDARY--\r\n"
	if want != buf.String() {
		t.Fatalf("\n got: %q\nwant: %q\n", buf.String(), want)
	}
}

func TestFileContentDisposition(t *testing.T) {
	tests := []struct {
		fieldname string
		filename  string
		want      string
	}{
		{"somefield", "somefile.txt", `form-data; name="somefield"; filename="somefile.txt"`},
		{`field"withquotes"`, "somefile.txt", `form-data; name="field\"withquotes\""; filename="somefile.txt"`},
		{`somefield`, `somefile"withquotes".txt`, `form-data; name="somefield"; filename="somefile\"withquotes\".txt"`},
		{`somefield\withbackslash`, "somefile.txt", `form-data; name="somefield\\withbackslash"; filename="somefile.txt"`},
		{"somefield", `somefile\withbackslash.txt`, `form-data; name="somefield"; filename="somefile\\withbackslash.txt"`},
	}
	for i, tt := range tests {
		if found := FileContentDisposition(tt.fieldname, tt.filename); found != tt.want {
			t.Errorf(`%d. found: "%s"; want: "%s"`, i, found, tt.want)
		}
	}
}

func TestFieldContentDisposition(t *testing.T) {
	tests := []struct {
		fieldname string
		want      string
	}{
		{"somefield", `form-data; name="somefield"`},
		{`field"withquotes"`, `form-data; name="field\"withquotes\""`},
		{`somefield\withbackslash`, `form-data; name="somefield\\withbackslash"`},
	}
	for i, tt := range tests {
		if found := FieldContentDisposition(tt.fieldname); found != tt.want {
			t.Errorf(`%d. found: "%s"; want: "%s"`, i, found, tt.want)
		}
	}
}
