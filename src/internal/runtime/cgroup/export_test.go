// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgroup

type LineReader = lineReader

func (l *LineReader) Next() error {
	return l.next()
}

func (l *LineReader) Line() []byte {
	return l.line()
}

func NewLineReader(fd int, scratch []byte, read func(fd int, b []byte) (int, uintptr)) *LineReader {
	return newLineReader(fd, scratch, read)
}

var (
	ErrEOF            = errEOF
	ErrIncompleteLine = errIncompleteLine
)
