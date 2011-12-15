// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go1

// Not a benchmark; input for revcomp.

var fasta25m = fasta(25e6)

func fasta(n int) []byte {
	out := make(fastaBuffer, 0, 11*n)

	iub := []fastaAcid{
		{prob: 0.27, sym: 'a'},
		{prob: 0.12, sym: 'c'},
		{prob: 0.12, sym: 'g'},
		{prob: 0.27, sym: 't'},
		{prob: 0.02, sym: 'B'},
		{prob: 0.02, sym: 'D'},
		{prob: 0.02, sym: 'H'},
		{prob: 0.02, sym: 'K'},
		{prob: 0.02, sym: 'M'},
		{prob: 0.02, sym: 'N'},
		{prob: 0.02, sym: 'R'},
		{prob: 0.02, sym: 'S'},
		{prob: 0.02, sym: 'V'},
		{prob: 0.02, sym: 'W'},
		{prob: 0.02, sym: 'Y'},
	}

	homosapiens := []fastaAcid{
		{prob: 0.3029549426680, sym: 'a'},
		{prob: 0.1979883004921, sym: 'c'},
		{prob: 0.1975473066391, sym: 'g'},
		{prob: 0.3015094502008, sym: 't'},
	}

	alu := []byte(
		"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG" +
			"GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA" +
			"CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT" +
			"ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA" +
			"GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG" +
			"AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC" +
			"AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA")

	out.WriteString(">ONE Homo sapiens alu\n")
	fastaRepeat(&out, alu, 2*n)
	out.WriteString(">TWO IUB ambiguity codes\n")
	fastaRandom(&out, iub, 3*n)
	out.WriteString(">THREE Homo sapiens frequency\n")
	fastaRandom(&out, homosapiens, 5*n)
	return out
}

type fastaBuffer []byte

func (b *fastaBuffer) Flush() {
	panic("flush")
}

func (b *fastaBuffer) WriteString(s string) {
	p := b.NextWrite(len(s))
	copy(p, s)
}

func (b *fastaBuffer) NextWrite(n int) []byte {
	p := *b
	if len(p)+n > cap(p) {
		b.Flush()
		p = *b
	}
	out := p[len(p) : len(p)+n]
	*b = p[:len(p)+n]
	return out
}

const fastaLine = 60

func fastaRepeat(out *fastaBuffer, alu []byte, n int) {
	buf := append(alu, alu...)
	off := 0
	for n > 0 {
		m := n
		if m > fastaLine {
			m = fastaLine
		}
		buf1 := out.NextWrite(m + 1)
		copy(buf1, buf[off:])
		buf1[m] = '\n'
		if off += m; off >= len(alu) {
			off -= len(alu)
		}
		n -= m
	}
}

const (
	fastaLookupSize          = 4096
	fastaLookupScale float64 = fastaLookupSize - 1
)

var fastaRand uint32 = 42

type fastaAcid struct {
	sym   byte
	prob  float64
	cprob float64
	next  *fastaAcid
}

func fastaComputeLookup(acid []fastaAcid) *[fastaLookupSize]*fastaAcid {
	var lookup [fastaLookupSize]*fastaAcid
	var p float64
	for i := range acid {
		p += acid[i].prob
		acid[i].cprob = p * fastaLookupScale
		if i > 0 {
			acid[i-1].next = &acid[i]
		}
	}
	acid[len(acid)-1].cprob = 1.0 * fastaLookupScale

	j := 0
	for i := range lookup {
		for acid[j].cprob < float64(i) {
			j++
		}
		lookup[i] = &acid[j]
	}

	return &lookup
}

func fastaRandom(out *fastaBuffer, acid []fastaAcid, n int) {
	const (
		IM = 139968
		IA = 3877
		IC = 29573
	)
	lookup := fastaComputeLookup(acid)
	for n > 0 {
		m := n
		if m > fastaLine {
			m = fastaLine
		}
		buf := out.NextWrite(m + 1)
		f := fastaLookupScale / IM
		myrand := fastaRand
		for i := 0; i < m; i++ {
			myrand = (myrand*IA + IC) % IM
			r := float64(int(myrand)) * f
			a := lookup[int(r)]
			for a.cprob < r {
				a = a.next
			}
			buf[i] = a.sym
		}
		fastaRand = myrand
		buf[m] = '\n'
		n -= m
	}
}
