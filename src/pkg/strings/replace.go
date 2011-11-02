// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import "io"

// A Replacer replaces a list of strings with replacements.
type Replacer struct {
	r replacer
}

// replacer is the interface that a replacement algorithm needs to implement.
type replacer interface {
	Replace(s string) string
	WriteString(w io.Writer, s string) (n int, err error)
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

	// Possible implementations.
	var (
		bb  byteReplacer
		bs  byteStringReplacer
		gen genericReplacer
	)

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

		// generic
		gen.p = append(gen.p, pair{old, new})

		// byte -> string
		if allOldBytes {
			bs.old.set(old[0])
			bs.new[old[0]] = []byte(new)
		}

		// byte -> byte
		if allOldBytes && allNewBytes {
			bb.old.set(old[0])
			bb.new[old[0]] = new[0]
		}
	}

	if allOldBytes && allNewBytes {
		return &Replacer{r: &bb}
	}
	if allOldBytes {
		return &Replacer{r: &bs}
	}
	return &Replacer{r: &gen}
}

// Replace returns a copy of s with all replacements performed.
func (r *Replacer) Replace(s string) string {
	return r.r.Replace(s)
}

// WriteString writes s to w with all replacements performed.
func (r *Replacer) WriteString(w io.Writer, s string) (n int, err error) {
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

func (w *appendSliceWriter) Write(p []byte) (int, error) {
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

func (r *genericReplacer) WriteString(w io.Writer, s string) (n int, err error) {
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

func (r *byteReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	// TODO(bradfitz): use io.WriteString with slices of s, avoiding allocation.
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

// byteStringReplacer is the implementation that's used when all the
// "old" values are single ASCII bytes but the "new" values vary in
// size.
type byteStringReplacer struct {
	// old has a bit set for each old byte that should be replaced.
	old byteBitmap

	// replacement string, indexed by old byte. only valid if
	// corresponding old bit is set.
	new [256][]byte
}

func (r *byteStringReplacer) Replace(s string) string {
	newSize := 0
	anyChanges := false
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			anyChanges = true
			newSize += len(r.new[b])
		} else {
			newSize++
		}
	}
	if !anyChanges {
		return s
	}
	buf := make([]byte, newSize)
	bi := buf
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			n := copy(bi[:], r.new[b])
			bi = bi[n:]
		} else {
			bi[0] = b
			bi = bi[1:]
		}
	}
	return string(buf)
}

// WriteString maintains one buffer that's at most 32KB.  The bytes in
// s are enumerated and the buffer is filled.  If it reaches its
// capacity or a byte has a replacement, the buffer is flushed to w.
func (r *byteStringReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	// TODO(bradfitz): use io.WriteString with slices of s instead.
	bufsize := 32 << 10
	if len(s) < bufsize {
		bufsize = len(s)
	}
	buf := make([]byte, bufsize)
	bi := buf[:0]

	for i := 0; i < len(s); i++ {
		b := s[i]
		var new []byte
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			new = r.new[b]
		} else {
			bi = append(bi, b)
		}
		if len(bi) == cap(bi) || (len(bi) > 0 && len(new) > 0) {
			nw, err := w.Write(bi)
			n += nw
			if err != nil {
				return n, err
			}
			bi = buf[:0]
		}
		if len(new) > 0 {
			nw, err := w.Write(new)
			n += nw
			if err != nil {
				return n, err
			}
		}
	}
	if len(bi) > 0 {
		nw, err := w.Write(bi)
		n += nw
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// strings is too low-level to import io/ioutil
var discard io.Writer = devNull(0)

type devNull int

func (devNull) Write(p []byte) (int, error) {
	return len(p), nil
}
