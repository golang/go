// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"testing"
)

func TestJSONFilterRewritePackage(t *testing.T) {
	const in = `{"Package":"abc"}
{"Field1":"1","Package":"abc","Field3":"3"}
{"Package":123}
{}
{"Package":"abc","Unexpected":[null,true,false,99999999999999999999]}
`
	want := strings.ReplaceAll(in, `"Package":"abc"`, `"Package":"abc:variant"`)

	checkJSONFilter(t, in, want)
}

func TestJSONFilterMalformed(t *testing.T) {
	const in = `unexpected text
{"Package":"abc"}
more text
{"Package":"abc"}trailing text
{not json}
no newline`
	const want = `unexpected text
{"Package":"abc:variant"}
more text
{"Package":"abc:variant"}trailing text
{not json}
no newline`
	checkJSONFilter(t, in, want)
}

func TestJSONFilterBoundaries(t *testing.T) {
	const in = `{"Package":"abc"}
{"Package":"def"}
{"Package":"ghi"}
`
	want := strings.ReplaceAll(in, `"}`, `:variant"}`)

	// Write one bytes at a time.
	t.Run("bytes", func { t ->
		checkJSONFilterWith(t, want, func { f -> for i := 0; i < len(in); i++ {
			f.Write([]byte{in[i]})
		} })
	})
	// Write a block containing a whole line bordered by two partial lines.
	t.Run("bytes", func { t ->
		checkJSONFilterWith(t, want, func { f ->
			const b1 = 5
			const b2 = len(in) - 5
			f.Write([]byte(in[:b1]))
			f.Write([]byte(in[b1:b2]))
			f.Write([]byte(in[b2:]))
		})
	})
}

func checkJSONFilter(t *testing.T, in, want string) {
	t.Helper()
	checkJSONFilterWith(t, want, func { f -> f.Write([]byte(in)) })
}

func checkJSONFilterWith(t *testing.T, want string, write func(*testJSONFilter)) {
	t.Helper()

	out := new(strings.Builder)
	f := &testJSONFilter{w: out, variant: "variant"}
	write(f)
	f.Flush()
	got := out.String()
	if want != got {
		t.Errorf("want:\n%s\ngot:\n%s", want, got)
	}
}
