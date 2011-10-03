// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"io"
	"os"
)

// A Replacer replaces a list of strings with replacements.
type Replacer struct {
	r replacer
}

// replacer is the interface that a replacement algorithm needs to implement.
type replacer interface {
	Replace(s string) string
	WriteString(w io.Writer, s string) (n int, err os.Error)
}

// byteBitmap represents bytes which are sought for replacement.
// byteBitmap is 256 bits wide, with a bit set for each old byte to be
// replaced.
type byteBitmap [256 / 32]uint32

func (m *byteBitmap) set(b byte) {
	m[b>>5] |= uint32(1 << (b & 31))
}

// NewReplacer returns a new Replacer from a list of old, new string pairs.
// Replacements are performed in order, without overlapping matches.
func NewReplacer(oldnew ...string) *Replacer {
	if len(oldnew)%2 == 1 {
		panic("strings.NewReplacer: odd argument count")
	}

	var bb byteReplacer
	var gen genericReplacer

	allOldBytes, allNewBytes := true, true
	for len(oldnew) > 0 {
		old, new := oldnew[0], oldnew[1]
		oldnew = oldnew[2:]
		if len(old) != 1 {
			allOldBytes = false
		}
		if len(new) != 1 {
			allNewBytes = false
		}
		gen.p = append(gen.p, pair{old, new})
		if allOldBytes && allNewBytes {
			bb.old.set(old[0])
			bb.new[old[0]] = new[0]
		}
	}

	if allOldBytes && allNewBytes {
		return &Replacer{r: &bb}
	}
	return &Replacer{r: &gen}
}

// Replace returns a copy of s with all replacements performed.
func (r *Replacer) Replace(s string) string {
	return r.r.Replace(s)
}

// WriteString writes s to w with all replacements performed.
func (r *Replacer) WriteString(w io.Writer, s string) (n int, err os.Error) {
	return r.r.WriteString(w, s)
}

// genericReplacer is the fully generic (and least optimized) algorithm.
// It's used as a fallback when nothing faster can be used.
type genericReplacer struct {
	p []pair
}

type pair struct{ old, new string }

type appendSliceWriter struct {
	b []byte
}

func (w *appendSliceWriter) Write(p []byte) (int, os.Error) {
	w.b = append(w.b, p...)
	return len(p), nil
}

func (r *genericReplacer) Replace(s string) string {
	// TODO(bradfitz): optimized version
	n, _ := r.WriteString(discard, s)
	w := appendSliceWriter{make([]byte, 0, n)}
	r.WriteString(&w, s)
	return string(w.b)
}

func (r *genericReplacer) WriteString(w io.Writer, s string) (n int, err os.Error) {
	lastEmpty := false // the last replacement was of the empty string
Input:
	// TODO(bradfitz): optimized version
	for i := 0; i < len(s); {
		for _, p := range r.p {
			if p.old == "" && lastEmpty {
				// Don't let old match twice in a row.
				// (it doesn't advance the input and
				// would otherwise loop forever)
				continue
			}
			if HasPrefix(s[i:], p.old) {
				if p.new != "" {
					wn, err := w.Write([]byte(p.new))
					n += wn
					if err != nil {
						return n, err
					}
				}
				i += len(p.old)
				lastEmpty = p.old == ""
				continue Input
			}
		}
		wn, err := w.Write([]byte{s[i]})
		n += wn
		if err != nil {
			return n, err
		}
		i++
	}

	// Final empty match at end.
	for _, p := range r.p {
		if p.old == "" {
			if p.new != "" {
				wn, err := w.Write([]byte(p.new))
				n += wn
				if err != nil {
					return n, err
				}
			}
			break
		}
	}

	return n, nil
}

// byteReplacer is the implementation that's used when all the "old"
// and "new" values are single ASCII bytes.
type byteReplacer struct {
	// old has a bit set for each old byte that should be replaced.
	old byteBitmap

	// replacement byte, indexed by old byte. only valid if
	// corresponding old bit is set.
	new [256]byte
}

func (r *byteReplacer) Replace(s string) string {
	var buf []byte // lazily allocated
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			if buf == nil {
				buf = []byte(s)
			}
			buf[i] = r.new[b]
		}
	}
	if buf == nil {
		return s
	}
	return string(buf)
}

func (r *byteReplacer) WriteString(w io.Writer, s string) (n int, err os.Error) {
	bufsize := 32 << 10
	if len(s) < bufsize {
		bufsize = len(s)
	}
	buf := make([]byte, bufsize)

	for len(s) > 0 {
		ncopy := copy(buf, s[:])
		s = s[ncopy:]
		for i, b := range buf[:ncopy] {
			if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
				buf[i] = r.new[b]
			}
		}
		wn, err := w.Write(buf[:ncopy])
		n += wn
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// strings is too low-level to import io/ioutil
var discard io.Writer = devNull(0)

type devNull int

func (devNull) Write(p []byte) (int, os.Error) {
	return len(p), nil
}
