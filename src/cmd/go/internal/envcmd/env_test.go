// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || windows

package envcmd

import (
	"bytes"
	"cmd/go/internal/cfg"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"unicode"
)

func FuzzPrintEnvEscape(f *testing.F) {
	f.Add(`$(echo 'cc"'; echo 'OOPS="oops')`)
	f.Add("$(echo shell expansion 1>&2)")
	f.Add("''")
	f.Add(`C:\"Program Files"\`)
	f.Add(`\\"Quoted Host"\\share`)
	f.Add("\xfb")
	f.Add("0")
	f.Add("")
	f.Add("''''''''")
	f.Add("\r")
	f.Add("\n")
	f.Add("E,%")
	f.Fuzz(func(t *testing.T, s string) {
		t.Parallel()

		for _, c := range []byte(s) {
			if c == 0 {
				t.Skipf("skipping %q: contains a null byte. Null bytes can't occur in the environment"+
					" outside of Plan 9, which has different code path than Windows and Unix that this test"+
					" isn't testing.", s)
			}
			if c > unicode.MaxASCII {
				t.Skipf("skipping %#q: contains a non-ASCII character %q", s, c)
			}
			if !unicode.IsGraphic(rune(c)) && !unicode.IsSpace(rune(c)) {
				t.Skipf("skipping %#q: contains non-graphic character %q", s, c)
			}
			if runtime.GOOS == "windows" && c == '\r' || c == '\n' {
				t.Skipf("skipping %#q on Windows: contains unescapable character %q", s, c)
			}
		}

		var b bytes.Buffer
		if runtime.GOOS == "windows" {
			b.WriteString("@echo off\n")
		}
		PrintEnv(&b, []cfg.EnvVar{{Name: "var", Value: s}}, false)
		var want string
		if runtime.GOOS == "windows" {
			fmt.Fprintf(&b, "echo \"%%var%%\"\n")
			want += "\"" + s + "\"\r\n"
		} else {
			fmt.Fprintf(&b, "printf '%%s\\n' \"$var\"\n")
			want += s + "\n"
		}
		scriptfilename := "script.sh"
		if runtime.GOOS == "windows" {
			scriptfilename = "script.bat"
		}
		var cmd *exec.Cmd
		if runtime.GOOS == "windows" {
			scriptfile := filepath.Join(t.TempDir(), scriptfilename)
			if err := os.WriteFile(scriptfile, b.Bytes(), 0777); err != nil {
				t.Fatal(err)
			}
			cmd = testenv.Command(t, "cmd.exe", "/C", scriptfile)
		} else {
			cmd = testenv.Command(t, "sh", "-c", b.String())
		}
		out, err := cmd.Output()
		t.Log(string(out))
		if err != nil {
			t.Fatal(err)
		}

		if string(out) != want {
			t.Fatalf("output of running PrintEnv script and echoing variable: got: %q, want: %q",
				string(out), want)
		}
	})
}
