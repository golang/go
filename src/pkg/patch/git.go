// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package patch

import (
	"bytes"
	"compress/zlib"
	"crypto/sha1"
	"encoding/git85"
	"errors"
	"fmt"
	"io"
)

func gitSHA1(data []byte) []byte {
	if len(data) == 0 {
		// special case: 0 length is all zeros sum
		return make([]byte, 20)
	}
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", len(data))
	h.Write(data)
	return h.Sum(nil)
}

// BUG(rsc): The Git binary delta format is not implemented, only Git binary literals.

// GitBinaryLiteral represents a Git binary literal diff.
type GitBinaryLiteral struct {
	OldSHA1 []byte // if non-empty, the SHA1 hash of the original
	New     []byte // the new contents
}

// Apply implements the Diff interface's Apply method.
func (d *GitBinaryLiteral) Apply(old []byte) ([]byte, error) {
	if sum := gitSHA1(old); !bytes.HasPrefix(sum, d.OldSHA1) {
		return nil, ErrPatchFailure
	}
	return d.New, nil
}

func unhex(c byte) uint8 {
	switch {
	case '0' <= c && c <= '9':
		return c - '0'
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10
	}
	return 255
}

func getHex(s []byte) (data []byte, rest []byte) {
	n := 0
	for n < len(s) && unhex(s[n]) != 255 {
		n++
	}
	n &^= 1 // Only take an even number of hex digits.
	data = make([]byte, n/2)
	for i := range data {
		data[i] = unhex(s[2*i])<<4 | unhex(s[2*i+1])
	}
	rest = s[n:]
	return
}

// ParseGitBinary parses raw as a Git binary patch.
func ParseGitBinary(raw []byte) (Diff, error) {
	var oldSHA1, newSHA1 []byte
	var sawBinary bool

	for {
		var first []byte
		first, raw, _ = getLine(raw, 1)
		first = bytes.TrimSpace(first)
		if s, ok := skip(first, "index "); ok {
			oldSHA1, s = getHex(s)
			if s, ok = skip(s, ".."); !ok {
				continue
			}
			newSHA1, s = getHex(s)
			continue
		}
		if _, ok := skip(first, "GIT binary patch"); ok {
			sawBinary = true
			continue
		}
		if n, _, ok := atoi(first, "literal ", 10); ok && sawBinary {
			data := make([]byte, n)
			d := git85.NewDecoder(bytes.NewBuffer(raw))
			z, err := zlib.NewReader(d)
			if err != nil {
				return nil, err
			}
			defer z.Close()
			if _, err = io.ReadFull(z, data); err != nil {
				if err == io.EOF {
					err = io.ErrUnexpectedEOF
				}
				return nil, err
			}
			var buf [1]byte
			m, err := z.Read(buf[0:])
			if m != 0 || err != io.EOF {
				return nil, errors.New("Git binary literal longer than expected")
			}

			if sum := gitSHA1(data); !bytes.HasPrefix(sum, newSHA1) {
				return nil, errors.New("Git binary literal SHA1 mismatch")
			}
			return &GitBinaryLiteral{oldSHA1, data}, nil
		}
		if !sawBinary {
			return nil, errors.New("unexpected Git patch header: " + string(first))
		}
	}
	panic("unreachable")
}
