// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quotedprintable

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestQuotedPrintable(t *testing.T) {
	tests := []struct {
		in, want string
		err      interface{}
	}{
		{in: "", want: ""},
		{in: "foo bar", want: "foo bar"},
		{in: "foo bar=3D", want: "foo bar="},
		{in: "foo bar=\n", want: "foo bar"},
		{in: "foo bar\n", want: "foo bar\n"}, // somewhat lax.
		{in: "foo bar=0", want: "foo bar", err: io.ErrUnexpectedEOF},
		{in: "foo bar=ab", want: "foo bar", err: "multipart: invalid quoted-printable hex byte 0x61"},
		{in: "foo bar=0D=0A", want: "foo bar\r\n"},
		{in: " A B        \r\n C ", want: " A B\r\n C"},
		{in: " A B =\r\n C ", want: " A B  C"},
		{in: " A B =\n C ", want: " A B  C"}, // lax. treating LF as CRLF
		{in: "foo=\nbar", want: "foobar"},
		{in: "foo\x00bar", want: "foo", err: "multipart: invalid unescaped byte 0x00 in quoted-printable body"},
		{in: "foo bar\xff", want: "foo bar", err: "multipart: invalid unescaped byte 0xff in quoted-printable body"},

		// Equal sign.
		{in: "=3D30\n", want: "=30\n"},
		{in: "=00=FF0=\n", want: "\x00\xff0"},

		// Trailing whitespace
		{in: "foo  \n", want: "foo\n"},
		{in: "foo  \n\nfoo =\n\nfoo=20\n\n", want: "foo\n\nfoo \nfoo \n\n"},

		// Tests that we allow bare \n and \r through, despite it being strictly
		// not permitted per RFC 2045, Section 6.7 Page 22 bullet (4).
		{in: "foo\nbar", want: "foo\nbar"},
		{in: "foo\rbar", want: "foo\rbar"},
		{in: "foo\r\nbar", want: "foo\r\nbar"},

		// Different types of soft line-breaks.
		{in: "foo=\r\nbar", want: "foobar"},
		{in: "foo=\nbar", want: "foobar"},
		{in: "foo=\rbar", want: "foo", err: "multipart: invalid quoted-printable hex byte 0x0d"},
		{in: "foo=\r\r\r \nbar", want: "foo", err: `multipart: invalid bytes after =: "\r\r\r \n"`},

		// Example from RFC 2045:
		{in: "Now's the time =\n" + "for all folk to come=\n" + " to the aid of their country.",
			want: "Now's the time for all folk to come to the aid of their country."},
	}
	for _, tt := range tests {
		var buf bytes.Buffer
		_, err := io.Copy(&buf, NewReader(strings.NewReader(tt.in)))
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

func everySequence(base, alpha string, length int, fn func(string)) {
	if len(base) == length {
		fn(base)
		return
	}
	for i := 0; i < len(alpha); i++ {
		everySequence(base+alpha[i:i+1], alpha, length, fn)
	}
}

var useQprint = flag.Bool("qprint", false, "Compare against the 'qprint' program.")

var badSoftRx = regexp.MustCompile(`=([^\r\n]+?\n)|([^\r\n]+$)|(\r$)|(\r[^\n]+\n)|( \r\n)`)

func TestQPExhaustive(t *testing.T) {
	if *useQprint {
		_, err := exec.LookPath("qprint")
		if err != nil {
			t.Fatalf("Error looking for qprint: %v", err)
		}
	}

	var buf bytes.Buffer
	res := make(map[string]int)
	everySequence("", "0A \r\n=", 6, func(s string) {
		if strings.HasSuffix(s, "=") || strings.Contains(s, "==") {
			return
		}
		buf.Reset()
		_, err := io.Copy(&buf, NewReader(strings.NewReader(s)))
		if err != nil {
			errStr := err.Error()
			if strings.Contains(errStr, "invalid bytes after =:") {
				errStr = "invalid bytes after ="
			}
			res[errStr]++
			if strings.Contains(errStr, "invalid quoted-printable hex byte ") {
				if strings.HasSuffix(errStr, "0x20") && (strings.Contains(s, "=0 ") || strings.Contains(s, "=A ") || strings.Contains(s, "= ")) {
					return
				}
				if strings.HasSuffix(errStr, "0x3d") && (strings.Contains(s, "=0=") || strings.Contains(s, "=A=")) {
					return
				}
				if strings.HasSuffix(errStr, "0x0a") || strings.HasSuffix(errStr, "0x0d") {
					// bunch of cases; since whitespace at the end of a line before \n is removed.
					return
				}
			}
			if strings.Contains(errStr, "unexpected EOF") {
				return
			}
			if errStr == "invalid bytes after =" && badSoftRx.MatchString(s) {
				return
			}
			t.Errorf("decode(%q) = %v", s, err)
			return
		}
		if *useQprint {
			cmd := exec.Command("qprint", "-d")
			cmd.Stdin = strings.NewReader(s)
			stderr, err := cmd.StderrPipe()
			if err != nil {
				panic(err)
			}
			qpres := make(chan interface{}, 2)
			go func() {
				br := bufio.NewReader(stderr)
				s, _ := br.ReadString('\n')
				if s != "" {
					qpres <- errors.New(s)
					if cmd.Process != nil {
						// It can get stuck on invalid input, like:
						// echo -n "0000= " | qprint -d
						cmd.Process.Kill()
					}
				}
			}()
			go func() {
				want, err := cmd.Output()
				if err == nil {
					qpres <- want
				}
			}()
			select {
			case got := <-qpres:
				if want, ok := got.([]byte); ok {
					if string(want) != buf.String() {
						t.Errorf("go decode(%q) = %q; qprint = %q", s, want, buf.String())
					}
				} else {
					t.Logf("qprint -d(%q) = %v", s, got)
				}
			case <-time.After(5 * time.Second):
				t.Logf("qprint timeout on %q", s)
			}
		}
		res["OK"]++
	})
	var outcomes []string
	for k, v := range res {
		outcomes = append(outcomes, fmt.Sprintf("%v: %d", k, v))
	}
	sort.Strings(outcomes)
	got := strings.Join(outcomes, "\n")
	want := `OK: 21576
invalid bytes after =: 3397
multipart: invalid quoted-printable hex byte 0x0a: 1400
multipart: invalid quoted-printable hex byte 0x0d: 2700
multipart: invalid quoted-printable hex byte 0x20: 2490
multipart: invalid quoted-printable hex byte 0x3d: 440
unexpected EOF: 3122`
	if got != want {
		t.Errorf("Got:\n%s\nWant:\n%s", got, want)
	}
}
