// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"bytes"
	"errors"
	"fmt"
	"log"
	"regexp"
	"testing"
)

type errWriter struct {
	err error
}

func (w errWriter) Write([]byte) (int, error) {
	return 0, w.err
}

func TestWriteLogger(t *testing.T) {
	olw := log.Writer()
	olf := log.Flags()
	olp := log.Prefix()

	// Revert the original log settings before we exit.
	defer func() {
		log.SetFlags(olf)
		log.SetPrefix(olp)
		log.SetOutput(olw)
	}()

	lOut := new(bytes.Buffer)
	log.SetPrefix("lw: ")
	log.SetOutput(lOut)
	log.SetFlags(0)

	lw := new(bytes.Buffer)
	wl := NewWriteLogger("write:", lw)
	if _, err := wl.Write([]byte("Hello, World!")); err != nil {
		t.Fatalf("Unexpectedly failed to write: %v", err)
	}

	if g, w := lw.String(), "Hello, World!"; g != w {
		t.Errorf("WriteLogger mismatch\n\tgot:  %q\n\twant: %q", g, w)
	}
	wantLogWithHex := fmt.Sprintf("lw: write: %x\n", "Hello, World!")
	if g, w := lOut.String(), wantLogWithHex; g != w {
		t.Errorf("WriteLogger mismatch\n\tgot:  %q\n\twant: %q", g, w)
	}
}

func TestWriteLoggerError(t *testing.T) {
	prefix := "prefix"
	data := "hello, world"
	err := errors.New("write error")
	re := regexp.MustCompile(`\d{4}\/\d{2}\/\d{2}\s\d{2}:\d{2}:\d{2}\s` + prefix + `\s:\s` + fmt.Sprintf("%s", err))
	out := new(bytes.Buffer)
	log.SetOutput(out)

	w := errWriter{err: err}
	wl := NewWriteLogger("prefix", w)
	wl.Write([]byte(data))

	if re.MatchString(out.String()) == false {
		t.Errorf("No match on log output, got %q", out.String())
	}

}

var readLoggerTests = []struct {
	prefix string
	data   string
}{
	{"", "hello, world"},
	{"prefix", ""},
	{"", ""},
	{"prefix", "hello, world"},
}

type errReader struct {
	err error
}

func (r errReader) Read([]byte) (int, error) {
	return 0, r.err
}

func TestReadLogger(t *testing.T) {
	for i, tt := range readLoggerTests {
		re := regexp.MustCompile(`\d{4}\/\d{2}\/\d{2}\s\d{2}:\d{2}:\d{2}\s` + tt.prefix + `\s` + fmt.Sprintf("%x", tt.data))

		p := make([]byte, len(tt.data))
		out := new(bytes.Buffer)
		log.SetOutput(out)

		r := bytes.NewReader([]byte(tt.data))
		rl := NewReadLogger(tt.prefix, r)
		rl.Read(p)

		if re.MatchString(out.String()) == false {
			t.Errorf("%d: No match on log output, got %q", i, out.String())
		}
	}
}

func TestReadLoggerError(t *testing.T) {
	prefix := "prefix"
	data := "hello, world"
	err := errors.New("read error")
	re := regexp.MustCompile(`\d{4}\/\d{2}\/\d{2}\s\d{2}:\d{2}:\d{2}\s` + prefix + `\s:\s` + fmt.Sprintf("%s", err))

	p := make([]byte, len(data))
	out := new(bytes.Buffer)
	log.SetOutput(out)

	r := errReader{err: err}
	rl := NewReadLogger(prefix, r)
	rl.Read(p)

	if re.MatchString(out.String()) == false {
		t.Errorf("No match on log output, got %q", out.String())
	}
}
