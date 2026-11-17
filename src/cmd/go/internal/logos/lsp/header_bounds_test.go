// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"errors"
	"io"
	"strconv"
	"strings"
	"testing"
)

// stopReader identifies a read that a header bound must make unnecessary.
type stopReader struct{ reads int }

var errReadPastHeader = errors.New("read past oversized header")

func (r *stopReader) Read([]byte) (int, error) {
	r.reads++
	return 0, errReadPastHeader
}

func TestHeaderBoundsBeforeAnotherRead(t *testing.T) {
	for name, input := range map[string]string{
		"unterminated first line": strings.Repeat("x", maxHeaderBytes+1),
		"unterminated later line": "Content-Length: 2\r\n" + strings.Repeat("x", maxHeaderBytes),
		"exhausted line count":    "Content-Length: 2\r\n" + strings.Repeat("X-Pad: x\r\n", maxHeaderLines-1),
	} {
		t.Run(name, func(t *testing.T) {
			stop := new(stopReader)
			rd := newReader(io.MultiReader(strings.NewReader(input), stop))
			_, err := rd.read()
			if err == nil || !strings.Contains(err.Error(), "message header too large") {
				t.Fatalf("read = %v, want oversized-header error", err)
			}
			if stop.reads != 0 {
				t.Fatalf("read %d times past oversized header, want none", stop.reads)
			}
		})
	}
}

func TestHeaderByteLimit(t *testing.T) {
	const prefix = "Content-Length: 2\r\nX-Pad: "
	const suffix = "\r\n\r\n"
	for _, size := range []int{maxHeaderBytes - 1, maxHeaderBytes, maxHeaderBytes + 1} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			header := prefix + strings.Repeat("x", size-len(prefix)-len(suffix)) + suffix
			_, err := newReader(strings.NewReader(header + "{}")).read()
			if size > maxHeaderBytes {
				if err == nil || !strings.Contains(err.Error(), "message header too large") {
					t.Fatalf("read = %v, want oversized-header error", err)
				}
			} else if err != nil {
				t.Fatalf("valid %d-byte header failed: %v", size, err)
			}
		})
	}
}

func TestHeaderLineLimit(t *testing.T) {
	for _, lines := range []int{maxHeaderLines - 1, maxHeaderLines, maxHeaderLines + 1} {
		t.Run(strconv.Itoa(lines), func(t *testing.T) {
			header := "Content-Length: 2\r\n" + strings.Repeat("X-Pad: x\r\n", lines-2) + "\r\n"
			_, err := newReader(strings.NewReader(header + "{}")).read()
			if lines > maxHeaderLines {
				if err == nil || !strings.Contains(err.Error(), "message header too large") {
					t.Fatalf("read = %v, want oversized-header error", err)
				}
			} else if err != nil {
				t.Fatalf("valid %d-line header failed: %v", lines, err)
			}
		})
	}
}

func TestHeaderBoundsPreserveBodyAndNextFrame(t *testing.T) {
	const first = `{"jsonrpc":"2.0","id":1,"result":"line\nContent-Length: 3"}`
	const second = `{"jsonrpc":"2.0","id":2,"result":null}`
	frame := func(body string) string {
		return "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n" + body
	}
	rd := newReader(strings.NewReader(frame(first) + frame(second)))
	for _, wantID := range []string{"1", "2"} {
		m, err := rd.read()
		if err != nil {
			t.Fatal(err)
		}
		if string(m.ID) != wantID {
			t.Fatalf("message ID = %s, want %s", m.ID, wantID)
		}
	}
	if _, err := rd.read(); !errors.Is(err, io.EOF) {
		t.Fatalf("after final message: %v, want EOF", err)
	}
}
