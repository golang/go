// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bytes"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
	"math/rand"
	"testing"
	"time"
	"unicode/utf8"
)

func TestEncodeArgs(t *testing.T) {
	t.Parallel()
	tests := []struct {
		arg, want string
	}{
		{"", `""`},
		{"hello", "hello"},
		{"hello\n", "\"hello\n\""},
		{"hello\\", `"hello\\"`},
		{"hello\nthere", "\"hello\nthere\""},
		{"\\\n", "\"\\\\\n\""},
		{"hello world", `"hello world"`},
		{"hello\tthere", "\"hello\tthere\""},
		{`hello"there`, `"hello\"there"`},
		{"hello$there", `"hello\$there"`},
		{"hello`there", "\"hello\\`there\""},
		{"simple", "simple"},
	}
	for _, test := range tests {
		if got := encodeArg(test.arg); got != test.want {
			t.Errorf("encodeArg(%q) = %q, want %q", test.arg, got, test.want)
		}
	}
}

func TestEncodeDecode(t *testing.T) {
	t.Parallel()
	tests := []string{
		"",
		"hello",
		"hello\\there",
		"hello\nthere",
		"hello 中国",
		"hello \n中\\国",
		"hello$world",
		"hello`world",
		`hello"world`,
	}
	for _, arg := range tests {
		encoded := encodeArg(arg)
		args := objabi.ParseArgs([]byte(encoded))
		if len(args) != 1 || args[0] != arg {
			t.Errorf("ParseArgs(encodeArg(%q)) = %q (encoded: %q)", arg, args, encoded)
		}
	}
}

func TestEncodeDecodeFuzz(t *testing.T) {
	if testing.Short() {
		t.Skip("fuzz test is slow")
	}
	t.Parallel()

	nRunes := sys.ExecArgLengthLimit + 100
	rBuffer := make([]rune, nRunes)
	buf := bytes.NewBuffer([]byte(string(rBuffer)))

	seed := time.Now().UnixNano()
	t.Logf("rand seed: %v", seed)
	rng := rand.New(rand.NewSource(seed))

	for i := 0; i < 50; i++ {
		// Generate a random string of runes.
		buf.Reset()
		for buf.Len() < sys.ExecArgLengthLimit+1 {
			var r rune
			for {
				r = rune(rng.Intn(utf8.MaxRune + 1))
				if utf8.ValidRune(r) {
					break
				}
			}
			fmt.Fprintf(buf, "%c", r)
		}
		arg := buf.String()

		encoded := encodeArg(arg)
		args := objabi.ParseArgs([]byte(encoded))
		if len(args) != 1 || args[0] != arg {
			t.Errorf("[%d] ParseArgs(encodeArg(%q)) = %q [seed: %v]", i, arg, args, seed)
		}
	}
}
