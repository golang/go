// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package raw

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode"

	"internal/trace/tracev2"
	"internal/trace/version"
)

// TextReader parses a text format trace with only very basic validation
// into an event stream.
type TextReader struct {
	v     version.Version
	specs []tracev2.EventSpec
	names map[string]tracev2.EventType
	s     *bufio.Scanner
}

// NewTextReader creates a new reader for the trace text format.
func NewTextReader(r io.Reader) (*TextReader, error) {
	tr := &TextReader{s: bufio.NewScanner(r)}
	line, err := tr.nextLine()
	if err != nil {
		return nil, err
	}
	trace, line := readToken(line)
	if trace != "Trace" {
		return nil, fmt.Errorf("failed to parse header")
	}
	gover, line := readToken(line)
	ver, cut := strings.CutPrefix(gover, "Go1.")
	if !cut {
		return nil, fmt.Errorf("failed to parse header Go version")
	}
	rawv, err := strconv.ParseUint(ver, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse header Go version: %v", err)
	}
	v := version.Version(rawv)
	if !v.Valid() {
		return nil, fmt.Errorf("unknown or unsupported Go version 1.%d", v)
	}
	tr.v = v
	tr.specs = v.Specs()
	tr.names = tracev2.EventNames(tr.specs)
	for _, r := range line {
		if !unicode.IsSpace(r) {
			return nil, fmt.Errorf("encountered unexpected non-space at the end of the header: %q", line)
		}
	}
	return tr, nil
}

// Version returns the version of the trace that we're reading.
func (r *TextReader) Version() version.Version {
	return r.v
}

// ReadEvent reads and returns the next trace event in the text stream.
func (r *TextReader) ReadEvent() (Event, error) {
	line, err := r.nextLine()
	if err != nil {
		return Event{}, err
	}
	evStr, line := readToken(line)
	ev, ok := r.names[evStr]
	if !ok {
		return Event{}, fmt.Errorf("unidentified event: %s", evStr)
	}
	spec := r.specs[ev]
	args, err := readArgs(line, spec.Args)
	if err != nil {
		return Event{}, fmt.Errorf("reading args for %s: %v", evStr, err)
	}
	if spec.IsStack {
		len := int(args[1])
		for i := 0; i < len; i++ {
			line, err := r.nextLine()
			if err == io.EOF {
				return Event{}, fmt.Errorf("unexpected EOF while reading stack: args=%v", args)
			}
			if err != nil {
				return Event{}, err
			}
			frame, err := readArgs(line, frameFields)
			if err != nil {
				return Event{}, err
			}
			args = append(args, frame...)
		}
	}
	var data []byte
	if spec.HasData {
		line, err := r.nextLine()
		if err == io.EOF {
			return Event{}, fmt.Errorf("unexpected EOF while reading data for %s: args=%v", evStr, args)
		}
		if err != nil {
			return Event{}, err
		}
		data, err = readData(line)
		if err != nil {
			return Event{}, err
		}
	}
	return Event{
		Version: r.v,
		Ev:      ev,
		Args:    args,
		Data:    data,
	}, nil
}

func (r *TextReader) nextLine() (string, error) {
	for {
		if !r.s.Scan() {
			if err := r.s.Err(); err != nil {
				return "", err
			}
			return "", io.EOF
		}
		txt := r.s.Text()
		tok, _ := readToken(txt)
		if tok == "" {
			continue // Empty line or comment.
		}
		return txt, nil
	}
}

var frameFields = []string{"pc", "func", "file", "line"}

func readArgs(s string, names []string) ([]uint64, error) {
	var args []uint64
	for _, name := range names {
		arg, value, rest, err := readArg(s)
		if err != nil {
			return nil, err
		}
		if arg != name {
			return nil, fmt.Errorf("expected argument %q, but got %q", name, arg)
		}
		args = append(args, value)
		s = rest
	}
	for _, r := range s {
		if !unicode.IsSpace(r) {
			return nil, fmt.Errorf("encountered unexpected non-space at the end of an event: %q", s)
		}
	}
	return args, nil
}

func readArg(s string) (arg string, value uint64, rest string, err error) {
	var tok string
	tok, rest = readToken(s)
	if len(tok) == 0 {
		return "", 0, s, fmt.Errorf("no argument")
	}
	parts := strings.SplitN(tok, "=", 2)
	if len(parts) < 2 {
		return "", 0, s, fmt.Errorf("malformed argument: %q", tok)
	}
	arg = parts[0]
	value, err = strconv.ParseUint(parts[1], 10, 64)
	if err != nil {
		return arg, value, s, fmt.Errorf("failed to parse argument value %q for arg %q", parts[1], parts[0])
	}
	return
}

func readToken(s string) (token, rest string) {
	tkStart := -1
	for i, r := range s {
		if r == '#' {
			return "", ""
		}
		if !unicode.IsSpace(r) {
			tkStart = i
			break
		}
	}
	if tkStart < 0 {
		return "", ""
	}
	tkEnd := -1
	for i, r := range s[tkStart:] {
		if unicode.IsSpace(r) || r == '#' {
			tkEnd = i + tkStart
			break
		}
	}
	if tkEnd < 0 {
		return s[tkStart:], ""
	}
	return s[tkStart:tkEnd], s[tkEnd:]
}

func readData(line string) ([]byte, error) {
	parts := strings.SplitN(line, "=", 2)
	if len(parts) < 2 || strings.TrimSpace(parts[0]) != "data" {
		return nil, fmt.Errorf("malformed data: %q", line)
	}
	data, err := strconv.Unquote(strings.TrimSpace(parts[1]))
	if err != nil {
		return nil, fmt.Errorf("failed to parse data: %q: %v", line, err)
	}
	return []byte(data), nil
}
