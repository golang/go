// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test2json

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

var update = flag.Bool("update", false, "rewrite testdata/*.json files")

func TestGolden(t *testing.T) {
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
