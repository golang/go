// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package test2json implements conversion of test binary output to JSON.
// It is used by cmd/test2json and cmd/go.
//
// See the cmd/test2json documentation for details of the JSON encoding.
package test2json

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// Mode controls details of the conversion.
type Mode int

const (
	Timestamp Mode = 1 << iota // include Time in events
)

// event is the JSON struct we emit.
type event struct {
	Time    *time.Time `json:",omitempty"`
	Action  string
	Package string     `json:",omitempty"`
	Test    string     `json:",omitempty"`
	Elapsed *float64   `json:",omitempty"`
	Output  *textBytes `json:",omitempty"`
}

// textBytes is a hack to get JSON to emit a []byte as a string
// without actually copying it to a string.
// It implements encoding.TextMarshaler, which returns its text form as a []byte,
// and then json encodes that text form as a string (which was our goal).
type textBytes []byte

func (b textBytes) MarshalText() ([]byte, error) { return b, nil }

// A Converter holds the state of a test-to-JSON conversion.
// It implements io.WriteCloser; the caller writes test output in,
// and the converter writes JSON output to w.
type Converter struct {
	w        io.Writer  // JSON output stream
	pkg      string     // package to name in events
	mode     Mode       // mode bits
	start    time.Time  // time converter started
	testName string     // name of current test, for output attribution
	report   []*event   // pending test result reports (nested for subtests)
	result   string     // overall test result if seen
	input    lineBuffer // input buffer
	output   lineBuffer // output buffer
}

// inBuffer and outBuffer are the input and output buffer sizes.
// They're variables so that they can be reduced during testing.
//
// The input buffer needs to be able to hold any single test
// directive line we want to recognize, like:
//
//	<many spaces> --- PASS: very/nested/s/u/b/t/e/s/t
//
// If anyone reports a test directive line > 4k not working, it will
// be defensible to suggest they restructure their test or test names.
//
// The output buffer must be >= utf8.UTFMax, so that it can
// accumulate any single UTF8 sequence. Lines that fit entirely
// within the output buffer are emitted in single output events.
// Otherwise they are split into multiple events.
// The output buffer size therefore limits the size of the encoding
// of a single JSON output event. 1k seems like a reasonable balance
// between wanting to avoid splitting an output line and not wanting to
// generate enormous output events.
var (
	inBuffer  = 4096
	outBuffer = 1024
)

// NewConverter returns a "test to json" converter.
// Writes on the returned writer are written as JSON to w,
// with minimal delay.
//
// The writes to w are whole JSON events ending in \n,
// so that it is safe to run multiple tests writing to multiple converters
// writing to a single underlying output stream w.
// As long as the underlying output w can handle concurrent writes
// from multiple goroutines, the result will be a JSON stream
// describing the relative ordering of execution in all the concurrent tests.
//
// The mode flag adjusts the behavior of the converter.
// Passing ModeTime includes event timestamps and elapsed times.
//
// The pkg string, if present, specifies the import path to
// report in the JSON stream.
func NewConverter(w io.Writer, pkg string, mode Mode) *Converter {
	c := new(Converter)
	*c = Converter{
		w:     w,
		pkg:   pkg,
		mode:  mode,
		start: time.Now(),
		input: lineBuffer{
			b:    make([]byte, 0, inBuffer),
			line: c.handleInputLine,
			part: c.output.write,
		},
		output: lineBuffer{
			b:    make([]byte, 0, outBuffer),
			line: c.writeOutputEvent,
			part: c.writeOutputEvent,
		},
	}
	return c
}

// Write writes the test input to the converter.
func (c *Converter) Write(b []byte) (int, error) {
	c.input.write(b)
	return len(b), nil
}

// Exited marks the test process as having exited with the given error.
func (c *Converter) Exited(err error) {
	if err == nil {
		c.result = "pass"
	} else {
		c.result = "fail"
	}
}

var (
	// printed by test on successful run.
	bigPass = []byte("PASS\n")

	// printed by test after a normal test failure.
	bigFail = []byte("FAIL\n")

	// printed by 'go test' along with an error if the test binary terminates
	// with an error.
	bigFailErrorPrefix = []byte("FAIL\t")

	updates = [][]byte{
		[]byte("=== RUN   "),
		[]byte("=== PAUSE "),
		[]byte("=== CONT  "),
	}

	reports = [][]byte{
		[]byte("--- PASS: "),
		[]byte("--- FAIL: "),
		[]byte("--- SKIP: "),
		[]byte("--- BENCH: "),
	}

	fourSpace = []byte("    ")

	skipLinePrefix = []byte("?   \t")
	skipLineSuffix = []byte("\t[no test files]\n")
)

// handleInputLine handles a single whole test output line.
// It must write the line to c.output but may choose to do so
// before or after emitting other events.
func (c *Converter) handleInputLine(line []byte) {
	// Final PASS or FAIL.
	if bytes.Equal(line, bigPass) || bytes.Equal(line, bigFail) || bytes.HasPrefix(line, bigFailErrorPrefix) {
		c.flushReport(0)
		c.output.write(line)
		if bytes.Equal(line, bigPass) {
			c.result = "pass"
		} else {
			c.result = "fail"
		}
		return
	}

	// Special case for entirely skipped test binary: "?   \tpkgname\t[no test files]\n" is only line.
	// Report it as plain output but remember to say skip in the final summary.
	if bytes.HasPrefix(line, skipLinePrefix) && bytes.HasSuffix(line, skipLineSuffix) && len(c.report) == 0 {
		c.result = "skip"
	}

	// "=== RUN   "
	// "=== PAUSE "
	// "=== CONT  "
	actionColon := false
	origLine := line
	ok := false
	indent := 0
	for _, magic := range updates {
		if bytes.HasPrefix(line, magic) {
			ok = true
			break
		}
	}
	if !ok {
		// "--- PASS: "
		// "--- FAIL: "
		// "--- SKIP: "
		// "--- BENCH: "
		// but possibly indented.
		for bytes.HasPrefix(line, fourSpace) {
			line = line[4:]
			indent++
		}
		for _, magic := range reports {
			if bytes.HasPrefix(line, magic) {
				actionColon = true
				ok = true
				break
			}
		}
	}

	// Not a special test output line.
	if !ok {
		// Lookup the name of the test which produced the output using the
		// indentation of the output as an index into the stack of the current
		// subtests.
		// If the indentation is greater than the number of current subtests
		// then the output must have included extra indentation. We can't
		// determine which subtest produced this output, so we default to the
		// old behaviour of assuming the most recently run subtest produced it.
		if indent > 0 && indent <= len(c.report) {
			c.testName = c.report[indent-1].Test
		}
		c.output.write(origLine)
		return
	}

	// Parse out action and test name.
	i := 0
	if actionColon {
		i = bytes.IndexByte(line, ':') + 1
	}
	if i == 0 {
		i = len(updates[0])
	}
	action := strings.ToLower(strings.TrimSuffix(strings.TrimSpace(string(line[4:i])), ":"))
	name := strings.TrimSpace(string(line[i:]))

	e := &event{Action: action}
	if line[0] == '-' { // PASS or FAIL report
		// Parse out elapsed time.
		if i := strings.Index(name, " ("); i >= 0 {
			if strings.HasSuffix(name, "s)") {
				t, err := strconv.ParseFloat(name[i+2:len(name)-2], 64)
				if err == nil {
					if c.mode&Timestamp != 0 {
						e.Elapsed = &t
					}
				}
			}
			name = name[:i]
		}
		if len(c.report) < indent {
			// Nested deeper than expected.
			// Treat this line as plain output.
			c.output.write(origLine)
			return
		}
		// Flush reports at this indentation level or deeper.
		c.flushReport(indent)
		e.Test = name
		c.testName = name
		c.report = append(c.report, e)
		c.output.write(origLine)
		return
	}
	// === update.
	// Finish any pending PASS/FAIL reports.
	c.flushReport(0)
	c.testName = name

	if action == "pause" {
		// For a pause, we want to write the pause notification before
		// delivering the pause event, just so it doesn't look like the test
		// is generating output immediately after being paused.
		c.output.write(origLine)
	}
	c.writeEvent(e)
	if action != "pause" {
		c.output.write(origLine)
	}

	return
}

// flushReport flushes all pending PASS/FAIL reports at levels >= depth.
func (c *Converter) flushReport(depth int) {
	c.testName = ""
	for len(c.report) > depth {
		e := c.report[len(c.report)-1]
		c.report = c.report[:len(c.report)-1]
		c.writeEvent(e)
	}
}

// Close marks the end of the go test output.
// It flushes any pending input and then output (only partial lines at this point)
// and then emits the final overall package-level pass/fail event.
func (c *Converter) Close() error {
	c.input.flush()
	c.output.flush()
	if c.result != "" {
		e := &event{Action: c.result}
		if c.mode&Timestamp != 0 {
			dt := time.Since(c.start).Round(1 * time.Millisecond).Seconds()
			e.Elapsed = &dt
		}
		c.writeEvent(e)
	}
	return nil
}

// writeOutputEvent writes a single output event with the given bytes.
func (c *Converter) writeOutputEvent(out []byte) {
	c.writeEvent(&event{
		Action: "output",
		Output: (*textBytes)(&out),
	})
}

// writeEvent writes a single event.
// It adds the package, time (if requested), and test name (if needed).
func (c *Converter) writeEvent(e *event) {
	e.Package = c.pkg
	if c.mode&Timestamp != 0 {
		t := time.Now()
		e.Time = &t
	}
	if e.Test == "" {
		e.Test = c.testName
	}
	js, err := json.Marshal(e)
	if err != nil {
		// Should not happen - event is valid for json.Marshal.
		c.w.Write([]byte(fmt.Sprintf("testjson internal error: %v\n", err)))
		return
	}
	js = append(js, '\n')
	c.w.Write(js)
}

// A lineBuffer is an I/O buffer that reacts to writes by invoking
// input-processing callbacks on whole lines or (for long lines that
// have been split) line fragments.
//
// It should be initialized with b set to a buffer of length 0 but non-zero capacity,
// and line and part set to the desired input processors.
// The lineBuffer will call line(x) for any whole line x (including the final newline)
// that fits entirely in cap(b). It will handle input lines longer than cap(b) by
// calling part(x) for sections of the line. The line will be split at UTF8 boundaries,
// and the final call to part for a long line includes the final newline.
type lineBuffer struct {
	b    []byte       // buffer
	mid  bool         // whether we're in the middle of a long line
	line func([]byte) // line callback
	part func([]byte) // partial line callback
}

// write writes b to the buffer.
func (l *lineBuffer) write(b []byte) {
	for len(b) > 0 {
		// Copy what we can into b.
		m := copy(l.b[len(l.b):cap(l.b)], b)
		l.b = l.b[:len(l.b)+m]
		b = b[m:]

		// Process lines in b.
		i := 0
		for i < len(l.b) {
			j := bytes.IndexByte(l.b[i:], '\n')
			if j < 0 {
				if !l.mid {
					if j := bytes.IndexByte(l.b[i:], '\t'); j >= 0 {
						if isBenchmarkName(bytes.TrimRight(l.b[i:i+j], " ")) {
							l.part(l.b[i : i+j+1])
							l.mid = true
							i += j + 1
						}
					}
				}
				break
			}
			e := i + j + 1
			if l.mid {
				// Found the end of a partial line.
				l.part(l.b[i:e])
				l.mid = false
			} else {
				// Found a whole line.
				l.line(l.b[i:e])
			}
			i = e
		}

		// Whatever's left in l.b is a line fragment.
		if i == 0 && len(l.b) == cap(l.b) {
			// The whole buffer is a fragment.
			// Emit it as the beginning (or continuation) of a partial line.
			t := trimUTF8(l.b)
			l.part(l.b[:t])
			l.b = l.b[:copy(l.b, l.b[t:])]
			l.mid = true
		}

		// There's room for more input.
		// Slide it down in hope of completing the line.
		if i > 0 {
			l.b = l.b[:copy(l.b, l.b[i:])]
		}
	}
}

// flush flushes the line buffer.
func (l *lineBuffer) flush() {
	if len(l.b) > 0 {
		// Must be a line without a \n, so a partial line.
		l.part(l.b)
		l.b = l.b[:0]
	}
}

var benchmark = []byte("Benchmark")

// isBenchmarkName reports whether b is a valid benchmark name
// that might appear as the first field in a benchmark result line.
func isBenchmarkName(b []byte) bool {
	if !bytes.HasPrefix(b, benchmark) {
		return false
	}
	if len(b) == len(benchmark) { // just "Benchmark"
		return true
	}
	r, _ := utf8.DecodeRune(b[len(benchmark):])
	return !unicode.IsLower(r)
}

// trimUTF8 returns a length t as close to len(b) as possible such that b[:t]
// does not end in the middle of a possibly-valid UTF-8 sequence.
//
// If a large text buffer must be split before position i at the latest,
// splitting at position trimUTF(b[:i]) avoids splitting a UTF-8 sequence.
func trimUTF8(b []byte) int {
	// Scan backward to find non-continuation byte.
	for i := 1; i < utf8.UTFMax && i <= len(b); i++ {
		if c := b[len(b)-i]; c&0xc0 != 0x80 {
			switch {
			case c&0xe0 == 0xc0:
				if i < 2 {
					return len(b) - i
				}
			case c&0xf0 == 0xe0:
				if i < 3 {
					return len(b) - i
				}
			case c&0xf8 == 0xf0:
				if i < 4 {
					return len(b) - i
				}
			}
			break
		}
	}
	return len(b)
}
