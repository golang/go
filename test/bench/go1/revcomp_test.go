// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark, taken from the shootout, tests array indexing
// and array bounds elimination performance.

package go1

import (
	"bufio"
	"bytes"
	"io/ioutil"
	"testing"
)

var revCompTable = [256]uint8{
	'A': 'T', 'a': 'T',
	'C': 'G', 'c': 'G',
	'G': 'C', 'g': 'C',
	'T': 'A', 't': 'A',
	'U': 'A', 'u': 'A',
	'M': 'K', 'm': 'K',
	'R': 'Y', 'r': 'Y',
	'W': 'W', 'w': 'W',
	'S': 'S', 's': 'S',
	'Y': 'R', 'y': 'R',
	'K': 'M', 'k': 'M',
	'V': 'B', 'v': 'B',
	'H': 'D', 'h': 'D',
	'D': 'H', 'd': 'H',
	'B': 'V', 'b': 'V',
	'N': 'N', 'n': 'N',
}

func revcomp(data []byte) {
	in := bufio.NewReader(bytes.NewBuffer(data))
	out := ioutil.Discard
	buf := make([]byte, 1024*1024)
	line, err := in.ReadSlice('\n')
	for err == nil {
		out.Write(line)

		// Accumulate reversed complement in buf[w:]
		nchar := 0
		w := len(buf)
		for {
			line, err = in.ReadSlice('\n')
			if err != nil || line[0] == '>' {
				break
			}
			line = line[0 : len(line)-1]
			nchar += len(line)
			if len(line)+nchar/60+128 >= w {
				nbuf := make([]byte, len(buf)*5)
				copy(nbuf[len(nbuf)-len(buf):], buf)
				w += len(nbuf) - len(buf)
				buf = nbuf
			}

			// This loop is the bottleneck.
			for _, c := range line {
				w--
				buf[w] = revCompTable[c]
			}
		}

		// Copy down to beginning of buffer, inserting newlines.
		// The loop left room for the newlines and 128 bytes of padding.
		i := 0
		for j := w; j < len(buf); j += 60 {
			n := copy(buf[i:i+60], buf[j:])
			buf[i+n] = '\n'
			i += n + 1
		}
		out.Write(buf[0:i])
	}
}

func BenchmarkRevcomp25M(b *testing.B) {
	b.SetBytes(int64(len(fasta25m)))
	for i := 0; i < b.N; i++ {
		revcomp(fasta25m)
	}
}
