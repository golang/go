// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"

	"cmd/compile/internal/base"
)

type chunk struct{ out, orig, n int }

type otext struct {
	s      string
	chunks []chunk
}

func gen(s string) otext { return otext{s: s} }

func cat(parts ...otext) otext {
	if len(parts) == 1 {
		return parts[0]
	}
	var sb strings.Builder
	for _, p := range parts {
		sb.WriteString(p.s)
	}
	out := sb.String()
	if out == "" {
		return otext{}
	}
	var chunks []chunk
	off := 0
	for _, p := range parts {
		for _, c := range p.chunks {
			if c.n == 0 {
				continue
			}
			chunks = append(chunks, chunk{out: off + c.out, orig: c.orig, n: c.n})
		}
		off += len(p.s)
	}
	return otext{s: out, chunks: chunks}
}

func (t otext) slice(start, end int) otext {
	sub := t.s[start:end]
	if sub == "" {
		return otext{}
	}
	var chunks []chunk
	for _, c := range t.chunks {
		cs, ce := c.out, c.out+c.n
		if ce <= start || cs >= end {
			continue
		}
		if cs < start {
			c.orig += start - cs
			cs = start
		}
		if ce > end {
			ce = end
		}
		chunks = append(chunks, chunk{out: cs - start, orig: c.orig, n: ce - cs})
	}
	return otext{s: sub, chunks: chunks}
}

func (t otext) trim() otext {
	start, end := 0, len(t.s)
	for start < end && isSpace(t.s[start]) {
		start++
	}
	for end > start && isSpace(t.s[end-1]) {
		end--
	}
	return t.slice(start, end)
}

func (t otext) trimRight(cutset string) otext {
	end := len(t.s)
	for end > 0 && strings.IndexByte(cutset, t.s[end-1]) >= 0 {
		end--
	}
	return t.slice(0, end)
}

func (t *otext) checkChunks(orig []byte) {
	prevEnd := -1
	for i, c := range t.chunks {
		if c.n <= 0 || c.out < 0 || c.orig < 0 || c.out+c.n > len(t.s) || c.orig+c.n > len(orig) {
			base.Fatalf("logos: invalid origin chunk %d %+v for %d bytes of text", i, c, len(t.s))
		}
		if c.out < prevEnd {
			base.Fatalf("logos: overlapping origin chunks at %d in %q", c.out, t.s)
		}
		if t.s[c.out:c.out+c.n] != string(orig[c.orig:c.orig+c.n]) {
			base.Fatalf("logos: origin chunk %+v is not a verbatim copy", c)
		}
		prevEnd = c.out + c.n
	}
}

func substNul(t otext, lookup func(mark string) (otext, bool)) otext {
	if !strings.ContainsRune(t.s, 0) {
		return t
	}
	var sb strings.Builder
	var chunks []chunk
	prev := 0
	scan := 0
	out := 0
	flush := func(upto int) {
		if upto <= prev {
			return
		}
		sb.WriteString(t.s[prev:upto])
		for _, c := range t.slice(prev, upto).chunks {
			chunks = append(chunks, chunk{out: out + c.out, orig: c.orig, n: c.n})
		}
		out += upto - prev
	}
	for scan < len(t.s) {
		i := strings.IndexByte(t.s[scan:], 0)
		if i < 0 {
			break
		}
		i += scan
		j := strings.IndexByte(t.s[i+1:], 0)
		if j < 0 {
			break
		}
		j += i + 1
		if repl, ok := lookup(t.s[i : j+1]); ok {
			flush(i)
			sb.WriteString(repl.s)
			for _, c := range repl.chunks {
				chunks = append(chunks, chunk{out: out + c.out, orig: c.orig, n: c.n})
			}
			out += len(repl.s)
			prev = j + 1
		}
		scan = j + 1
	}
	if prev == 0 {
		return t
	}
	flush(len(t.s))
	res := otext{s: sb.String(), chunks: chunks}
	if res.s == "" {
		return otext{}
	}
	return res
}

func substNeedle(t otext, needle, repl otext) otext {
	if needle.s == "" || !strings.Contains(t.s, needle.s) {
		return t
	}
	var sb strings.Builder
	var chunks []chunk
	prev := 0
	out := 0
	emit := func(p otext) {
		sb.WriteString(p.s)
		for _, c := range p.chunks {
			chunks = append(chunks, chunk{out: out + c.out, orig: c.orig, n: c.n})
		}
		out += len(p.s)
	}
	for {
		i := strings.Index(t.s[prev:], needle.s)
		if i < 0 {
			break
		}
		i += prev
		emit(t.slice(prev, i))
		emit(repl)
		prev = i + len(needle.s)
	}
	emit(t.slice(prev, len(t.s)))
	res := otext{s: sb.String(), chunks: chunks}
	if res.s == "" {
		return otext{}
	}
	return res
}

type srcBuf struct {
	b      []byte
	chunks []chunk
}

func origBuf(b []byte) srcBuf {
	if len(b) == 0 {
		return srcBuf{}
	}
	return srcBuf{b: b, chunks: []chunk{{out: 0, orig: 0, n: len(b)}}}
}

func (b *srcBuf) sliceO(start, end int) otext {
	return otext{s: string(b.b[start:end]), chunks: b.clip(start, end)}
}

func (b *srcBuf) clip(start, end int) []chunk {
	if end <= start {
		return nil
	}
	var out []chunk
	for _, c := range b.chunks {
		cs, ce := c.out, c.out+c.n
		if ce <= start || cs >= end {
			continue
		}
		oc := c.orig
		if cs < start {
			oc += start - cs
			cs = start
		}
		if ce > end {
			ce = end
		}
		out = append(out, chunk{out: cs - start, orig: oc, n: ce - cs})
	}
	return out
}

func (b *srcBuf) appendO(t otext) {
	base := len(b.b)
	b.b = append(b.b, t.s...)
	for _, c := range t.chunks {
		if c.n == 0 {
			continue
		}
		b.chunks = append(b.chunks, chunk{out: base + c.out, orig: c.orig, n: c.n})
	}
}
