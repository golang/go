// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stack

import (
	"bufio"
	"errors"
	"io"
	"regexp"
	"strconv"
)

var (
	reBlank     = regexp.MustCompile(`^\s*$`)
	reGoroutine = regexp.MustCompile(`^\s*goroutine (\d+) \[([^\]]*)\]:\s*$`)
	reCall      = regexp.MustCompile(`^\s*` +
		`(created by )?` + //marker
		`(([\w/.]+/)?[\w]+)\.` + //package
		`(\(([^:.)]*)\)\.)?` + //optional type
		`([\w\.]+)` + //function
		`(\(.*\))?` + // args
		`\s*$`)
	rePos = regexp.MustCompile(`^\s*(.*):(\d+)( .*)?$`)

	errBreakParse = errors.New("break parse")
)

// Scanner splits an input stream into lines in a way that is consumable by
// the parser.
type Scanner struct {
	lines *bufio.Scanner
	done  bool
}

// NewScanner creates a scanner on top of a reader.
func NewScanner(r io.Reader) *Scanner {
	s := &Scanner{
		lines: bufio.NewScanner(r),
	}
	s.Skip() // prefill
	return s
}

// Peek returns the next line without consuming it.
func (s *Scanner) Peek() string {
	if s.done {
		return ""
	}
	return s.lines.Text()
}

// Skip consumes the next line without looking at it.
// Normally used after it has already been looked at using Peek.
func (s *Scanner) Skip() {
	if !s.lines.Scan() {
		s.done = true
	}
}

// Next consumes and returns the next line.
func (s *Scanner) Next() string {
	line := s.Peek()
	s.Skip()
	return line
}

// Done returns true if the scanner has reached the end of the underlying
// stream.
func (s *Scanner) Done() bool {
	return s.done
}

// Err returns true if the scanner has reached the end of the underlying
// stream.
func (s *Scanner) Err() error {
	return s.lines.Err()
}

// Match returns the submatchs of the regular expression against the next line.
// If it matched the line is also consumed.
func (s *Scanner) Match(re *regexp.Regexp) []string {
	if s.done {
		return nil
	}
	match := re.FindStringSubmatch(s.Peek())
	if match != nil {
		s.Skip()
	}
	return match
}

// SkipBlank skips any number of pure whitespace lines.
func (s *Scanner) SkipBlank() {
	for !s.done {
		line := s.Peek()
		if len(line) != 0 && !reBlank.MatchString(line) {
			return
		}
		s.Skip()
	}
}

// Parse the current contiguous block of goroutine stack traces until the
// scanned content no longer matches.
func Parse(scanner *Scanner) (Dump, error) {
	dump := Dump{}
	for {
		gr, ok := parseGoroutine(scanner)
		if !ok {
			return dump, nil
		}
		dump = append(dump, gr)
	}
}

func parseGoroutine(scanner *Scanner) (Goroutine, bool) {
	match := scanner.Match(reGoroutine)
	if match == nil {
		return Goroutine{}, false
	}
	id, _ := strconv.ParseInt(match[1], 0, 32)
	gr := Goroutine{
		ID:    int(id),
		State: match[2],
	}
	for {
		frame, ok := parseFrame(scanner)
		if !ok {
			scanner.SkipBlank()
			return gr, true
		}
		if frame.Position.Filename != "" {
			gr.Stack = append(gr.Stack, frame)
		}
	}
}

func parseFrame(scanner *Scanner) (Frame, bool) {
	fun, ok := parseFunction(scanner)
	if !ok {
		return Frame{}, false
	}
	frame := Frame{
		Function: fun,
	}
	frame.Position, ok = parsePosition(scanner)
	// if ok is false, then this is a broken state.
	// we got the func but not the file that must follow
	// the consumed line can be recovered from the frame
	//TODO: push back the fun raw
	return frame, ok
}

func parseFunction(scanner *Scanner) (Function, bool) {
	match := scanner.Match(reCall)
	if match == nil {
		return Function{}, false
	}
	return Function{
		Package: match[2],
		Type:    match[5],
		Name:    match[6],
	}, true
}

func parsePosition(scanner *Scanner) (Position, bool) {
	match := scanner.Match(rePos)
	if match == nil {
		return Position{}, false
	}
	line, _ := strconv.ParseInt(match[2], 0, 32)
	return Position{Filename: match[1], Line: int(line)}, true
}
