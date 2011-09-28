// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"io"
	"os"
)

// Can't import ioutil for ioutil.Discard, due to ioutil/tempfile.go -> strconv -> strings
var discard io.Writer = devNull(0)

type devNull int

func (devNull) Write(p []byte) (int, os.Error) {
	return len(p), nil
}

type pair struct{ old, new string }

// A Replacer replaces a list of strings with replacements.
type Replacer struct {
	p []pair
}

// NewReplacer returns a new Replacer from a list of old, new string pairs.
// Replacements are performed in order, without overlapping matches.
func NewReplacer(oldnew ...string) *Replacer {
	if len(oldnew)%2 == 1 {
		panic("strings.NewReplacer: odd argument count")
	}
	r := new(Replacer)
	for len(oldnew) >= 2 {
		r.p = append(r.p, pair{oldnew[0], oldnew[1]})
		oldnew = oldnew[2:]
	}
	return r
}

type appendSliceWriter struct {
	b []byte
}

func (w *appendSliceWriter) Write(p []byte) (int, os.Error) {
	w.b = append(w.b, p...)
	return len(p), nil
}

// Replace returns a copy of s with all replacements performed.
func (r *Replacer) Replace(s string) string {
	// TODO(bradfitz): optimized version
	n, _ := r.WriteString(discard, s)
	w := appendSliceWriter{make([]byte, 0, n)}
	r.WriteString(&w, s)
	return string(w.b)
}

// WriteString writes s to w with all replacements performed.
func (r *Replacer) WriteString(w io.Writer, s string) (n int, err os.Error) {
Input:
	// TODO(bradfitz): optimized version
	for i := 0; i < len(s); {
		for _, p := range r.p {
			if HasPrefix(s[i:], p.old) {
				wn, err := w.Write([]byte(p.new))
				n += wn
				if err != nil {
					return n, err
				}
				i += len(p.old)
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
	return n, nil
}
