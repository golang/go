// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test2json

import (
	"bufio"
	"bytes"
	"cmd/internal/script"
	"cmd/internal/script/scripttest"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"internal/txtar"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"
)

var update = flag.Bool("update", false, "rewrite testdata/*.json files")

func TestGolden(t *testing.T) {
	ctx := scripttest.ScriptTestContext(t, context.Background())
	engine, env := scripttest.NewEngine(t, nil)
	files, err := filepath.Glob("testdata/*.test")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		name := strings.TrimSuffix(filepath.Base(file), ".test")
		t.Run(name, func(t *testing.T) {
			orig, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}

			// If there's a corresponding *.src script, execute it
			srcFile := strings.TrimSuffix(file, ".test") + ".src"
			if st, err := os.Stat(srcFile); err != nil {
				if !errors.Is(err, fs.ErrNotExist) {
					t.Fatal(err)
				}
			} else if !st.IsDir() {
				t.Run("go test", func(t *testing.T) {
					stdout := runTest(t, ctx, engine, env, srcFile)

					if *update {
						t.Logf("rewriting %s", file)
						if err := os.WriteFile(file, []byte(stdout), 0666); err != nil {
							t.Fatal(err)
						}
						orig = []byte(stdout)
						return
					}

					diffRaw(t, []byte(stdout), orig)
				})
			}

			// Test one line written to c at a time.
			// Assume that's the most likely to be handled correctly.
			var buf bytes.Buffer
			c := NewConverter(&buf, "", 0)
			in := append([]byte{}, orig...)
			for _, line := range bytes.SplitAfter(in, []byte("\n")) {
				writeAndKill(c, line)
			}
			c.Close()

			if *update {
				js := strings.TrimSuffix(file, ".test") + ".json"
				t.Logf("rewriting %s", js)
				if err := os.WriteFile(js, buf.Bytes(), 0666); err != nil {
					t.Fatal(err)
				}
				return
			}

			want, err := os.ReadFile(strings.TrimSuffix(file, ".test") + ".json")
			if err != nil {
				t.Fatal(err)
			}
			diffJSON(t, buf.Bytes(), want)
			if t.Failed() {
				// If the line-at-a-time conversion fails, no point testing boundary conditions.
				return
			}

			// Write entire input in bulk.
			t.Run("bulk", func(t *testing.T) {
				buf.Reset()
				c = NewConverter(&buf, "", 0)
				in = append([]byte{}, orig...)
				writeAndKill(c, in)
				c.Close()
				diffJSON(t, buf.Bytes(), want)
			})

			// In bulk again with \r\n.
			t.Run("crlf", func(t *testing.T) {
				buf.Reset()
				c = NewConverter(&buf, "", 0)
				in = bytes.ReplaceAll(orig, []byte("\n"), []byte("\r\n"))
				writeAndKill(c, in)
				c.Close()
				diffJSON(t, bytes.ReplaceAll(buf.Bytes(), []byte(`\r\n`), []byte(`\n`)), want)
			})

			// Write 2 bytes at a time on even boundaries.
			t.Run("even2", func(t *testing.T) {
				buf.Reset()
				c = NewConverter(&buf, "", 0)
				in = append([]byte{}, orig...)
				for i := 0; i < len(in); i += 2 {
					if i+2 <= len(in) {
						writeAndKill(c, in[i:i+2])
					} else {
						writeAndKill(c, in[i:])
					}
				}
				c.Close()
				diffJSON(t, buf.Bytes(), want)
			})

			// Write 2 bytes at a time on odd boundaries.
			t.Run("odd2", func(t *testing.T) {
				buf.Reset()
				c = NewConverter(&buf, "", 0)
				in = append([]byte{}, orig...)
				if len(in) > 0 {
					writeAndKill(c, in[:1])
				}
				for i := 1; i < len(in); i += 2 {
					if i+2 <= len(in) {
						writeAndKill(c, in[i:i+2])
					} else {
						writeAndKill(c, in[i:])
					}
				}
				c.Close()
				diffJSON(t, buf.Bytes(), want)
			})

			// Test with very small output buffers, to check that
			// UTF8 sequences are not broken up.
			for b := 5; b <= 8; b++ {
				t.Run(fmt.Sprintf("tiny%d", b), func(t *testing.T) {
					oldIn := inBuffer
					oldOut := outBuffer
					defer func() {
						inBuffer = oldIn
						outBuffer = oldOut
					}()
					inBuffer = 64
					outBuffer = b
					buf.Reset()
					c = NewConverter(&buf, "", 0)
					in = append([]byte{}, orig...)
					writeAndKill(c, in)
					c.Close()
					diffJSON(t, buf.Bytes(), want)
				})
			}
		})
	}
}

func runTest(t *testing.T, ctx context.Context, engine *script.Engine, env []string, srcFile string) string {
	workdir := t.TempDir()
	s, err := script.NewState(ctx, workdir, env)
	if err != nil {
		t.Fatal(err)
	}

	// Unpack archive.
	a, err := txtar.ParseFile(srcFile)
	if err != nil {
		t.Fatal(err)
	}
	scripttest.InitScriptDirs(t, s)
	if err := s.ExtractFiles(a); err != nil {
		t.Fatal(err)
	}

	err, stdout := func() (err error, stdout string) {
		log := new(strings.Builder)

		// Defer writing to the test log in case the script engine panics during execution,
		// but write the log before we write the final "skip" or "FAIL" line.
		t.Helper()
		defer func() {
			t.Helper()

			stdout = s.Stdout()
			if closeErr := s.CloseAndWait(log); err == nil {
				err = closeErr
			}

			if log.Len() > 0 && (testing.Verbose() || err != nil) {
				t.Log(strings.TrimSuffix(log.String(), "\n"))
			}
		}()

		if testing.Verbose() {
			// Add the environment to the start of the script log.
			wait, err := script.Env().Run(s)
			if err != nil {
				t.Fatal(err)
			}
			if wait != nil {
				stdout, stderr, err := wait(s)
				if err != nil {
					t.Fatalf("env: %v\n%s", err, stderr)
				}
				if len(stdout) > 0 {
					s.Logf("%s\n", stdout)
				}
			}
		}

		testScript := bytes.NewReader(a.Comment)
		err = engine.Execute(s, srcFile, bufio.NewReader(testScript), log)
		return
	}()
	if skip := (scripttest.SkipError{}); errors.As(err, &skip) {
		t.Skipf("SKIP: %v", skip)
	} else if err != nil {
		t.Fatalf("FAIL: %v", err)
	}

	// Remove the output after "=== NAME"
	i := strings.LastIndex(stdout, "\n\x16=== NAME")
	if i >= 0 {
		stdout = stdout[:i+1]
	}

	return stdout
}

// writeAndKill writes b to w and then fills b with Zs.
// The filling makes sure that if w is holding onto b for
// future use, that future use will have obviously wrong data.
func writeAndKill(w io.Writer, b []byte) {
	w.Write(b)
	for i := range b {
		b[i] = 'Z'
	}
}

// diffJSON diffs the stream we have against the stream we want
// and fails the test with a useful message if they don't match.
func diffJSON(t *testing.T, have, want []byte) {
	t.Helper()
	type event map[string]any

	// Parse into events, one per line.
	parseEvents := func(b []byte) ([]event, []string) {
		t.Helper()
		var events []event
		var lines []string
		for _, line := range bytes.SplitAfter(b, []byte("\n")) {
			if len(line) > 0 {
				line = bytes.TrimSpace(line)
				var e event
				err := json.Unmarshal(line, &e)
				if err != nil {
					t.Errorf("unmarshal %s: %v", b, err)
					continue
				}
				events = append(events, e)
				lines = append(lines, string(line))
			}
		}
		return events, lines
	}
	haveEvents, haveLines := parseEvents(have)
	wantEvents, wantLines := parseEvents(want)
	if t.Failed() {
		return
	}

	// Make sure the events we have match the events we want.
	// At each step we're matching haveEvents[i] against wantEvents[j].
	// i and j can move independently due to choices about exactly
	// how to break up text in "output" events.
	i := 0
	j := 0

	// Fail reports a failure at the current i,j and stops the test.
	// It shows the events around the current positions,
	// with the current positions marked.
	fail := func() {
		var buf bytes.Buffer
		show := func(i int, lines []string) {
			for k := -2; k < 5; k++ {
				marker := ""
				if k == 0 {
					marker = "» "
				}
				if 0 <= i+k && i+k < len(lines) {
					fmt.Fprintf(&buf, "\t%s%s\n", marker, lines[i+k])
				}
			}
			if i >= len(lines) {
				// show marker after end of input
				fmt.Fprintf(&buf, "\t» \n")
			}
		}
		fmt.Fprintf(&buf, "have:\n")
		show(i, haveLines)
		fmt.Fprintf(&buf, "want:\n")
		show(j, wantLines)
		t.Fatal(buf.String())
	}

	var outputTest string             // current "Test" key in "output" events
	var wantOutput, haveOutput string // collected "Output" of those events

	// getTest returns the "Test" setting, or "" if it is missing.
	getTest := func(e event) string {
		s, _ := e["Test"].(string)
		return s
	}

	// checkOutput collects output from the haveEvents for the current outputTest
	// and then checks that the collected output matches the wanted output.
	checkOutput := func() {
		for i < len(haveEvents) && haveEvents[i]["Action"] == "output" && getTest(haveEvents[i]) == outputTest {
			haveOutput += haveEvents[i]["Output"].(string)
			i++
		}
		if haveOutput != wantOutput {
			t.Errorf("output mismatch for Test=%q:\nhave %q\nwant %q", outputTest, haveOutput, wantOutput)
			fail()
		}
		haveOutput = ""
		wantOutput = ""
	}

	// Walk through wantEvents matching against haveEvents.
	for j = range wantEvents {
		e := wantEvents[j]
		if e["Action"] == "output" && getTest(e) == outputTest {
			wantOutput += e["Output"].(string)
			continue
		}
		checkOutput()
		if e["Action"] == "output" {
			outputTest = getTest(e)
			wantOutput += e["Output"].(string)
			continue
		}
		if i >= len(haveEvents) {
			t.Errorf("early end of event stream: missing event")
			fail()
		}
		if !reflect.DeepEqual(haveEvents[i], e) {
			t.Errorf("events out of sync")
			fail()
		}
		i++
	}
	checkOutput()
	if i < len(haveEvents) {
		t.Errorf("extra events in stream")
		fail()
	}
}

var reRuntime = regexp.MustCompile(`\d*\.\d*s`)

func diffRaw(t *testing.T, have, want []byte) {
	have = bytes.TrimSpace(have)
	want = bytes.TrimSpace(want)

	// Replace durations (e.g. 0.01s) with a placeholder
	have = reRuntime.ReplaceAll(have, []byte("X.XXs"))
	want = reRuntime.ReplaceAll(want, []byte("X.XXs"))

	// Compare
	if bytes.Equal(have, want) {
		return
	}

	// Escape non-printing characters to make the error more legible
	have = escapeNonPrinting(have)
	want = escapeNonPrinting(want)

	// Find where the output differs and remember the last newline
	var i, nl int
	for i < len(have) && i < len(want) && have[i] == want[i] {
		if have[i] == '\n' {
			nl = i
		}
	}

	if nl == 0 {
		t.Fatalf("\nhave:\n%s\nwant:\n%s", have, want)
	} else {
		nl++
		t.Fatalf("\nhave:\n%s» %s\nwant:\n%s» %s", have[:nl], have[nl:], want[:nl], want[nl:])
	}
}

func escapeNonPrinting(buf []byte) []byte {
	for i := 0; i < len(buf); i++ {
		c := buf[i]
		if 0x20 <= c && c < 0x7F || c > 0x7F || c == '\n' {
			continue
		}
		escaped := fmt.Sprintf(`\x%02x`, c)
		buf = append(buf[:i+len(escaped)], buf[i+1:]...)
		for j := 0; j < len(escaped); j++ {
			buf[i+j] = escaped[j]
		}
	}
	return buf
}

func TestTrimUTF8(t *testing.T) {
	s := "hello α ☺ 😂 world" // α is 2-byte, ☺ is 3-byte, 😂 is 4-byte
	b := []byte(s)
	for i := 0; i < len(s); i++ {
		j := trimUTF8(b[:i])
		u := string([]rune(s[:j])) + string([]rune(s[j:]))
		if u != s {
			t.Errorf("trimUTF8(%q) = %d (-%d), not at boundary (split: %q %q)", s[:i], j, i-j, s[:j], s[j:])
		}
		if utf8.FullRune(b[j:i]) {
			t.Errorf("trimUTF8(%q) = %d (-%d), too early (missed: %q)", s[:j], j, i-j, s[j:i])
		}
	}
}
