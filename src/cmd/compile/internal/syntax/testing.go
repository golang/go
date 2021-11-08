// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements testing support.

package syntax

import (
	"io"
	"regexp"
	"strings"
)

// CommentsDo parses the given source and calls the provided handler for each
// comment or error. If the text provided to handler starts with a '/' it is
// the comment text; otherwise it is the error message.
func CommentsDo(src io.Reader, handler func(line, col uint, text string)) {
	var s scanner
	s.init(src, handler, comments)
	for s.tok != _EOF {
		s.next()
	}
}

// ERROR comments must start with text `ERROR "msg"` or `ERROR msg`.
// Space around "msg" or msg is ignored.
var errRx = regexp.MustCompile(`^ *ERROR *"?([^"]*)"?`)

// ErrorMap collects all comments with comment text of the form
// `ERROR "msg"` or `ERROR msg` from the given src and returns them
// as []Error lists in a map indexed by line number. The position
// for each Error is the position of the token immediately preceding
// the comment, the Error message is the message msg extracted from
// the comment, with all errors that are on the same line collected
// in a slice, in source order. If there is no preceding token (the
// `ERROR` comment appears in the beginning of the file), then the
// recorded position is unknown (line, col = 0, 0). If there are no
// ERROR comments, the result is nil.
func ErrorMap(src io.Reader) (errmap map[uint][]Error) {
	// position of previous token
	var base *PosBase
	var prev struct{ line, col uint }

	var s scanner
	s.init(src, func(_, _ uint, text string) {
		if text[0] != '/' {
			return // error, ignore
		}
		if text[1] == '*' {
			text = text[:len(text)-2] // strip trailing */
		}
		if s := errRx.FindStringSubmatch(text[2:]); len(s) == 2 {
			pos := MakePos(base, prev.line, prev.col)
			err := Error{pos, strings.TrimSpace(s[1])}
			if errmap == nil {
				errmap = make(map[uint][]Error)
			}
			errmap[prev.line] = append(errmap[prev.line], err)
		}
	}, comments)

	for s.tok != _EOF {
		s.next()
		if s.tok == _Semi && s.lit != "semicolon" {
			continue // ignore automatically inserted semicolons
		}
		prev.line, prev.col = s.line, s.col
	}

	return
}
