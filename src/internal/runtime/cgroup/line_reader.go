// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgroup

import (
	"internal/bytealg"
)

// stringError is a trival implementation of error, equivalent to errors.New,
// which cannot be imported from a runtime package.
type stringError string

func (e stringError) Error() string {
	return string(e)
}

// All errors are explicit converted to type error in global initialization to
// ensure that the linker allocates a static interface value. This is necessary
// because these errors may be used before the allocator is available.

var (
	// The entire line did not fit into the scratch buffer.
	errIncompleteLine error = stringError("incomplete line")

	// A system call failed.
	errSyscallFailed error = stringError("syscall failed")

	// Reached EOF.
	errEOF error = stringError("end of file")
)

// lineReader reads line-by-line using only a single fixed scratch buffer.
//
// When a single line is too long for the scratch buffer, the remainder of the
// line will be skipped.
type lineReader struct {
	read    func(fd int, b []byte) (int, uintptr)
	fd      int
	scratch []byte

	n       int // bytes of scratch in use.
	newline int // index of the first newline in scratch.

	eof bool // read reached EOF.
}

// newLineReader returns a lineReader which reads lines from fd.
//
// fd is the file descriptor to read from.
//
// scratch is the scratch buffer to read into. Note that len(scratch) is the
// longest line that can be read. Lines longer than len(scratch) will have the
// remainder of the line skipped. See next for more details.
//
// read is the function used to read more bytes from fd. This is usually
// internal/runtime/syscall.Read. Note that this follows syscall semantics (not
// io.Reader), so EOF is indicated with n=0, errno=0.
func newLineReader(fd int, scratch []byte, read func(fd int, b []byte) (n int, errno uintptr)) *lineReader {
	return &lineReader{
		read:    read,
		fd:      fd,
		scratch: scratch,
		n:       0,
		newline: -1,
	}
}

// next advances to the next line.
//
// May return errIncompleteLine if the scratch buffer is too small to hold the
// entire line, in which case [r.line] will return the beginning of the line. A
// subsequent call to next will skip the remainder of the incomplete line.
//
// N.B. this behavior is important for /proc/self/mountinfo. Some lines
// (mounts), such as overlayfs, may be extremely long due to long super-block
// options, but we don't care about those. The mount type will appear early in
// the line.
//
// Returns errEOF when there are no more lines.
func (r *lineReader) next() error {
	// Three cases:
	//
	// 1. First call, no data read.
	// 2. Previous call had a complete line. Drop it and look for the end
	//    of the next line.
	// 3. Previous call had an incomplete line. Find the end of that line
	//    (start of the next line), and the end of the next line.

	prevComplete := r.newline >= 0
	firstCall := r.n == 0

	for {
		if prevComplete {
			// Drop the previous line.
			copy(r.scratch, r.scratch[r.newline+1:r.n])
			r.n -= r.newline + 1

			r.newline = bytealg.IndexByte(r.scratch[:r.n], '\n')
			if r.newline >= 0 {
				// We have another line already in scratch. Done.
				return nil
			}
		}

		// No newline available.

		if !prevComplete {
			// If the previous line was incomplete, we are
			// searching for the end of that line and have no need
			// for any buffered data.
			r.n = 0
		}

		n, errno := r.read(r.fd, r.scratch[r.n:len(r.scratch)])
		if errno != 0 {
			return errSyscallFailed
		}
		r.n += n

		if r.n == 0 {
			// Nothing left.
			//
			// N.B. we can't immediately return EOF when read
			// returns 0 as we may still need to return an
			// incomplete line.
			return errEOF
		}

		r.newline = bytealg.IndexByte(r.scratch[:r.n], '\n')
		if prevComplete || firstCall {
			// Already have the start of the line, just need to find the end.

			if r.newline < 0 {
				// We filled the entire buffer or hit EOF, but
				// still no newline.
				return errIncompleteLine
			}

			// Found the end of the line. Done.
			return nil
		} else {
			// Don't have the start of the line. We are currently
			// looking for the end of the previous line.

			if r.newline < 0 {
				// Not there yet.
				if n == 0 {
					// No more to read.
					return errEOF
				}
				continue
			}

			// Found the end of the previous line. The next
			// iteration will drop the remainder of the previous
			// line and look for the next line.
			prevComplete = true
		}
	}
}

// line returns a view of the current line, excluding the trailing newline.
//
// If [r.next] returned errIncompleteLine, then this returns only the beginning
// of the line.
//
// Preconditions: [r.next] is called prior to the first call to line.
//
// Postconditions: The caller must not keep a reference to the returned slice.
func (r *lineReader) line() []byte {
	if r.newline < 0 {
		// Incomplete line
		return r.scratch[:r.n]
	}
	// Complete line.
	return r.scratch[:r.newline]
}
