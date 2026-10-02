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
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestToolexecLongArgsResponseFile(t *testing.T) {
	tool := filepath.Join(t.TempDir(), "compile")
	wrapper := filepath.Join(t.TempDir(), "wrapper")
	args := []string{tool, "-o", "out file", strings.Repeat("source.go", sys.ExecArgLengthLimit/8)}
	cmd := exec.Command(wrapper, args...)
	cleanup := passLongArgsInResponseFiles(cmd, 1)
	defer cleanup()
	if len(cmd.Args) != 3 || cmd.Args[1] != tool || !strings.HasPrefix(cmd.Args[2], "@") {
		t.Fatalf("toolexec args = %q, want wrapper, tool, @response-file", cmd.Args)
	}
	content, err := os.ReadFile(strings.TrimPrefix(cmd.Args[2], "@"))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := objabi.ParseArgs(content), args[1:]; !slices.Equal(got, want) {
		t.Fatalf("response-file args = %q, want %q", got, want)
	}
}

func TestToolexecShortArgsUnchanged(t *testing.T) {
	wrapper := filepath.Join(t.TempDir(), "wrapper")
	tool := filepath.Join(t.TempDir(), "compile")
	cmd := exec.Command(wrapper, "-verbose", tool, "-o", "out")
	cleanup := passLongArgsInResponseFiles(cmd, 2)
	defer cleanup()
	if got, want := cmd.Args, []string{wrapper, "-verbose", tool, "-o", "out"}; !slices.Equal(got, want) {
		t.Fatalf("short toolexec args = %q, want %q", got, want)
	}
}

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
