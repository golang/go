// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package work

import (
	"bytes"
	"internal/testenv"
	"strings"
	"testing"
	"unicode"
)

func FuzzSplitPkgConfigOutput(f *testing.F) {
	testenv.MustHaveExecPath(f, "/bin/sh")

	f.Add([]byte(`$FOO`))
	f.Add([]byte(`\$FOO`))
	f.Add([]byte(`${FOO}`))
	f.Add([]byte(`\${FOO}`))
	f.Add([]byte(`$(/bin/false)`))
	f.Add([]byte(`\$(/bin/false)`))
	f.Add([]byte(`$((0))`))
	f.Add([]byte(`\$((0))`))
	f.Add([]byte(`unescaped space`))
	f.Add([]byte(`escaped\ space`))
	f.Add([]byte(`"unterminated quote`))
	f.Add([]byte(`'unterminated quote`))
	f.Add([]byte(`unterminated escape\`))
	f.Add([]byte(`"quote with unterminated escape\`))
	f.Add([]byte(`'quoted "double quotes"'`))
	f.Add([]byte(`"quoted 'single quotes'"`))
	f.Add([]byte(`"\$0"`))
	f.Add([]byte(`"\$\0"`))
	f.Add([]byte(`"\$"`))
	f.Add([]byte(`"\$ "`))

	// Example positive inputs from TestSplitPkgConfigOutput.
	// Some bare newlines have been removed so that the inputs
	// are valid in the shell script we use for comparison.
	f.Add([]byte(`-r:foo -L/usr/white\ space/lib -lfoo\ bar -lbar\ baz`))
	f.Add([]byte(`-lextra\ fun\ arg\\`))
	f.Add([]byte("\textra     whitespace\r"))
	f.Add([]byte("     \r      "))
	f.Add([]byte(`"-r:foo" "-L/usr/white space/lib" "-lfoo bar" "-lbar baz"`))
	f.Add([]byte(`"-lextra fun arg\\"`))
	f.Add([]byte(`"     \r\n\      "`))
	f.Add([]byte(`""`))
	f.Add([]byte(``))
	f.Add([]byte(`"\\"`))
	f.Add([]byte(`"\x"`))
	f.Add([]byte(`"\\x"`))
	f.Add([]byte(`'\\'`))
	f.Add([]byte(`'\x'`))
	f.Add([]byte(`"\\x"`))
	f.Add([]byte("\\\n"))
	f.Add([]byte(`-fPIC -I/test/include/foo -DQUOTED='"/test/share/doc"'`))
	f.Add([]byte(`-fPIC -I/test/include/foo -DQUOTED="/test/share/doc"`))
	f.Add([]byte(`-fPIC -I/test/include/foo -DQUOTED=\"/test/share/doc\"`))
	f.Add([]byte(`-fPIC -I/test/include/foo -DQUOTED='/test/share/doc'`))
	f.Add([]byte(`-DQUOTED='/te\st/share/d\oc'`))
	f.Add([]byte(`-Dhello=10 -Dworld=+32 -DDEFINED_FROM_PKG_CONFIG=hello\ world`))
	f.Add([]byte(`"broken\"" \\\a "a"`))

	// Example negative inputs from TestSplitPkgConfigOutput.
	f.Add([]byte(`"     \r\n      `))
	f.Add([]byte(`"-r:foo" "-L/usr/white space/lib "-lfoo bar" "-lbar baz"`))
	f.Add([]byte(`"-lextra fun arg\\`))
	f.Add([]byte(`broken flag\`))
	f.Add([]byte(`extra broken flag \`))
	f.Add([]byte(`\`))
	f.Add([]byte(`"broken\"" "extra" \`))

	f.Fuzz(func(t *testing.T, b []byte) {
		t.Parallel()

		if bytes.ContainsAny(b, "*?[#~%\x00{}!") {
			t.Skipf("skipping %#q: contains a sometimes-quoted character", b)
		}
		// splitPkgConfigOutput itself rejects inputs that contain unquoted
		// shell operator characters. (Quoted shell characters are fine.)

		for _, c := range b {
			if c > unicode.MaxASCII {
				t.Skipf("skipping %#q: contains a non-ASCII character %q", b, c)
			}
			if !unicode.IsGraphic(rune(c)) && !unicode.IsSpace(rune(c)) {
				t.Skipf("skipping %#q: contains non-graphic character %q", b, c)
			}
		}

		args, err := splitPkgConfigOutput(b)
		if err != nil {
			// We haven't checked that the shell would actually reject this input too,
			// but if splitPkgConfigOutput rejected it it's probably too dangerous to
			// run in the script.
			t.Logf("%#q: %v", b, err)
			return
		}
		t.Logf("splitPkgConfigOutput(%#q) = %#q", b, args)
		if len(args) == 0 {
			t.Skipf("skipping %#q: contains no arguments", b)
		}

		var buf strings.Builder
		for _, arg := range args {
			buf.WriteString(arg)
			buf.WriteString("\n")
		}
		wantOut := buf.String()

		if strings.Count(wantOut, "\n") != len(args)+bytes.Count(b, []byte("\n")) {
			// One of the newlines in b was treated as a delimiter and not part of an
			// argument. Our bash test script would interpret that as a syntax error.
			t.Skipf("skipping %#q: contains a bare newline", b)
		}

		// We use the printf shell command to echo the arguments because, per
		// https://pubs.opengroup.org/onlinepubs/9699919799/utilities/echo.html#tag_20_37_16:
		// “It is not possible to use echo portably across all POSIX systems unless
		// both -n (as the first argument) and escape sequences are omitted.”
		cmd := testenv.Command(t, "/bin/sh", "-c", "printf '%s\n' "+string(b))
		cmd.Env = append(cmd.Environ(), "LC_ALL=POSIX", "POSIXLY_CORRECT=1")
		cmd.Stderr = new(strings.Builder)
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("%#q: %v\n%s", cmd.Args, err, cmd.Stderr)
		}

		if string(out) != wantOut {
			t.Logf("%#q:\n%#q", cmd.Args, out)
			t.Logf("want:\n%#q", wantOut)
			t.Errorf("parsed args do not match")
		}
	})
}
