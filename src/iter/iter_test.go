// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iter_test

import (
	"bufio"
	"errors"
	"fmt"
	"iter"
	"os"
	"path/filepath"
)

func Example() {

	// errorHandler prints error message and exits 1 on error
	var err error
	defer errorHandler(&err)

	// create test file
	var filename = filepath.Join(os.TempDir(), "test.txt")
	if err = os.WriteFile(filename, []byte("one\ntwo\n"), 0o600); err != nil {
		return
	}

	// iterate over lines from test.txt
	//   - the LineReader iterator is allocated on the stack
	//   - stack allocation is faster than heap allocation
	//   - LineReader is on stack even if NewLineReader is in another module
	//   - LineReader pointer receiver is more performant
	for line := range NewLineReader(&LineReader{}, filename, &err).Lines {
		fmt.Println("iterator line:", line)
	}
	// return here, err may be non-nil

	// Output:
	// iterator line: one
	// iterator line: two
}

// LineReader provides an iterator reading a file line-by-line
type LineReader struct {
	// the file lines are being read from
	filename string
	// a pointer to store occurring errors
	errp *error
	// the open file
	osFile *os.File
}

// NewLineReader returns an iterator over the lines of a file
//   - [LineReader.Lines] is iterator function
//   - new-function provides LineReader encapsulation
func NewLineReader(fieldp *LineReader, filename string, errp *error) (lineReader *LineReader) {
	if fieldp != nil {
		lineReader = fieldp
		osFile = nil
	} else {
		lineReader = &LineReader{}
	}
	lineReader.filename = filename
	lineReader.errp = errp

	return
}

// Lines is the iterator providing text-lines from the file filename
//   - defer cleanup ensures cleanup is executed on panic
//     in Lines method or for block
//   - cleanup updates *LineReader.errp
func (r *LineReader) Lines(yield func(line string) (keepGoing bool)) {
	var err error
	defer r.cleanup(&err)

	if r.osFile, err = os.Open(r.filename); err != nil {
		return // i/o error
	}
	var scanner = bufio.NewScanner(r.osFile)
	for scanner.Scan() {
		if !yield(scanner.Text()) {
			return // iteration canceled by break or such
		}
	}
	// reached end of file
}

// LineReader.Lines is iter.Seq string
var _ iter.Seq[string] = (&LineReader{}).Lines

// cleanup is invoked on iteration end or any panic
//   - errp: possible error from Lines
func (r *LineReader) cleanup(errp *error) {
	var err error
	if r.osFile != nil {
		err = r.osFile.Close()
	}
	if err != nil || *errp != nil {
		// aggregate errors in order of occurrence
		*r.errp = errors.Join(*r.errp, *errp, err)
	}
}

// errorHandler prints error message and exits 1 on error
//   - deferrable
func errorHandler(errp *error) {
	var err = *errp
	if err == nil {
		return
	}
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
