// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tlog

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// A Tree is a tree description, to be signed by a go.sum database server.
type Tree struct {
	N    int64
	Hash Hash
}

// FormatTree formats a tree description for inclusion in a note.
//
// The encoded form is three lines, each ending in a newline (U+000A):
//
//	go.sum database tree
//	N
//	Hash
//
// where N is in decimal and Hash is in base64.
//
// A future backwards-compatible encoding may add additional lines,
// which the parser can ignore.
// A future backwards-incompatible encoding would use a different
// first line (for example, "go.sum database tree v2").
func FormatTree(tree Tree) []byte {
	return fmt.Appendf(nil, "go.sum database tree\n%d\n%s\n", tree.N, tree.Hash)
}

var errMalformedTree = errors.New("malformed tree note")
var treePrefix = []byte("go.sum database tree\n")

// ParseTree parses a formatted tree root description.
func ParseTree(text []byte) (tree Tree, err error) {
	// The message looks like:
	//
	//	go.sum database tree
	//	2
	//	nND/nri/U0xuHUrYSy0HtMeal2vzD9V4k/BO79C+QeI=
	//
	// For forwards compatibility, extra text lines after the encoding are ignored.
	if !bytes.HasPrefix(text, treePrefix) || bytes.Count(text, []byte("\n")) < 3 || len(text) > 1e6 {
		return Tree{}, errMalformedTree
	}

	lines := strings.SplitN(string(text), "\n", 4)
	n, err := strconv.ParseInt(lines[1], 10, 64)
	if err != nil || n < 0 || lines[1] != strconv.FormatInt(n, 10) {
		return Tree{}, errMalformedTree
	}

	h, err := base64.StdEncoding.DecodeString(lines[2])
	if err != nil || len(h) != HashSize {
		return Tree{}, errMalformedTree
	}

	var hash Hash
	copy(hash[:], h)
	return Tree{n, hash}, nil
}

var errMalformedRecord = errors.New("malformed record data")

// FormatRecord formats a record for serving to a client
// in a lookup response.
//
// The encoded form is the record ID as a single number,
// then the text of the record, and then a terminating blank line.
// Record text must be valid UTF-8 and must not contain any ASCII control
// characters (those below U+0020) other than newline (U+000A).
// It must end in a terminating newline and not contain any blank lines.
//
// Responses to data tiles consist of concatenated formatted records from each of
// which the first line, with the record ID, is removed.
func FormatRecord(id int64, text []byte) (msg []byte, err error) {
	if !isValidRecordText(text) {
		return nil, errMalformedRecord
	}
	msg = fmt.Appendf(nil, "%d\n", id)
	msg = append(msg, text...)
	msg = append(msg, '\n')
	return msg, nil
}

// isValidRecordText reports whether text is syntactically valid record text.
func isValidRecordText(text []byte) bool {
	var last rune
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRune(text[i:])
		if r < 0x20 && r != '\n' || r == utf8.RuneError && size == 1 || last == '\n' && r == '\n' {
			return false
		}
		i += size
		last = r
	}
	if last != '\n' {
		return false
	}
	return true
}

// ParseRecord parses a record description at the start of text,
// stopping immediately after the terminating blank line.
// It returns the record id, the record text, and the remainder of text.
func ParseRecord(msg []byte) (id int64, text, rest []byte, err error) {
	// Leading record id.
	i := bytes.IndexByte(msg, '\n')
	if i < 0 {
		return 0, nil, nil, errMalformedRecord
	}
	id, err = strconv.ParseInt(string(msg[:i]), 10, 64)
	if err != nil {
		return 0, nil, nil, errMalformedRecord
	}
	msg = msg[i+1:]

	// Record text.
	i = bytes.Index(msg, []byte("\n\n"))
	if i < 0 {
		return 0, nil, nil, errMalformedRecord
	}
	text, rest = msg[:i+1], msg[i+2:]
	if !isValidRecordText(text) {
		return 0, nil, nil, errMalformedRecord
	}
	return id, text, rest, nil
}
