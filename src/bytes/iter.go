// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes

import (
	"iter"
	"unicode"
	"unicode/utf8"
)

// Lines returns an iterator over the newline-terminated lines in the byte slice s.
// The lines yielded by the iterator include their terminating newlines.
// If s is empty, the iterator yields no lines at all.
// If s does not end in a newline, the final yielded line will not end in a newline.
// It returns a single-use iterator.
func Lines(s []byte) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		for len(s) > 0 {
			var line []byte
			if i := IndexByte(s, '\n'); i >= 0 {
				line, s = s[:i+1], s[i+1:]
			} else {
				line, s = s, nil
			}
			if !yield(line[:len(line):len(line)]) {
				return
			}
		}
		return
	}
}

// explodeSeq returns an iterator over the runes in s.
func explodeSeq(s []byte) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		for len(s) > 0 {
			_, size := utf8.DecodeRune(s)
			if !yield(s[:size:size]) {
				return
			}
			s = s[size:]
		}
	}
}

// splitSeq is SplitSeq or SplitAfterSeq, configured by how many
// bytes of sep to include in the results (none or all).
func splitSeq(s, sep []byte, sepSave int) iter.Seq[[]byte] {
	if len(sep) == 0 {
		return explodeSeq(s)
	}
	return func(yield func([]byte) bool) {
		for {
			i := Index(s, sep)
			if i < 0 {
				break
			}
			frag := s[:i+sepSave]
			if !yield(frag[:len(frag):len(frag)]) {
				return
			}
			s = s[i+len(sep):]
		}
		yield(s[:len(s):len(s)])
	}
}

// SplitSeq returns an iterator over all subslices of s separated by sep.
// The iterator yields the same subslices that would be returned by [Split](s, sep),
// but without constructing a new slice containing the subslices.
// It returns a single-use iterator.
func SplitSeq(s, sep []byte) iter.Seq[[]byte] {
	return splitSeq(s, sep, 0)
}

// SplitAfterSeq returns an iterator over subslices of s split after each instance of sep.
// The iterator yields the same subslices that would be returned by [SplitAfter](s, sep),
// but without constructing a new slice containing the subslices.
// It returns a single-use iterator.
func SplitAfterSeq(s, sep []byte) iter.Seq[[]byte] {
	return splitSeq(s, sep, len(sep))
}

// FieldsSeq returns an iterator over subslices of s split around runs of
// whitespace characters, as defined by [unicode.IsSpace].
// The iterator yields the same subslices that would be returned by [Fields](s),
// but without constructing a new slice containing the subslices.
func FieldsSeq(s []byte) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		start := -1
		for i := 0; i < len(s); {
			size := 1
			r := rune(s[i])
			isSpace := asciiSpace[s[i]] != 0
			if r >= utf8.RuneSelf {
				r, size = utf8.DecodeRune(s[i:])
				isSpace = unicode.IsSpace(r)
			}
			if isSpace {
				if start >= 0 {
					if !yield(s[start:i:i]) {
						return
					}
					start = -1
				}
			} else if start < 0 {
				start = i
			}
			i += size
		}
		if start >= 0 {
			yield(s[start:len(s):len(s)])
		}
	}
}

// FieldsFuncSeq returns an iterator over subslices of s split around runs of
// Unicode code points satisfying f(c).
// The iterator yields the same subslices that would be returned by [FieldsFunc](s),
// but without constructing a new slice containing the subslices.
func FieldsFuncSeq(s []byte, f func(rune) bool) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		start := -1
		for i := 0; i < len(s); {
			size := 1
			r := rune(s[i])
			if r >= utf8.RuneSelf {
				r, size = utf8.DecodeRune(s[i:])
			}
			if f(r) {
				if start >= 0 {
					if !yield(s[start:i:i]) {
						return
					}
					start = -1
				}
			} else if start < 0 {
				start = i
			}
			i += size
		}
		if start >= 0 {
			yield(s[start:len(s):len(s)])
		}
	}
}
