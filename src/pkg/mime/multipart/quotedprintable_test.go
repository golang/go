// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestQuotedPrintable(t *testing.T) {
	tests := []struct {
		in, want string
		err      interface{}
	}{
		{in: "foo bar", want: "foo bar"},
		{in: "foo bar=3D", want: "foo bar="},
		{in: "foo bar=0", want: "foo bar", err: io.ErrUnexpectedEOF},
		{in: "foo bar=ab", want: "foo bar", err: "multipart: invalid quoted-printable hex byte 0x61"},
		{in: "foo bar=0D=0A", want: "foo bar\r\n"},
		{in: "foo bar=\r\n baz", want: "foo bar baz"},
		{in: "foo=\nbar", want: "foobar"},
		{in: "foo\x00bar", want: "foo", err: "multipart: invalid unescaped byte 0x00 in quoted-printable body"},
		{in: "foo bar\xff", want: "foo bar", err: "multipart: invalid unescaped byte 0xff in quoted-printable body"},
	}
	for _, tt := range tests {
		var buf bytes.Buffer
		_, err := io.Copy(&buf, newQuotedPrintableReader(strings.NewReader(tt.in)))
		if got := buf.String(); got != tt.want {
			t.Errorf("for %q, got %q; want %q", tt.in, got, tt.want)
		}
		switch verr := tt.err.(type) {
		case nil:
			if err != nil {
				t.Errorf("for %q, got unexpected error: %v", tt.in, err)
			}
		case string:
			if got := fmt.Sprint(err); got != verr {
				t.Errorf("for %q, got error %q; want %q", tt.in, got, verr)
			}
		case error:
			if err != verr {
				t.Errorf("for %q, got error %q; want %q", tt.in, err, verr)
			}
		}
	}

}
