// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package block

import (
	"bytes"
	"fmt"
	"io"
	"testing"
	"testing/iotest"
)

// Simple "pseudo-random" stream for testing.
type incStream struct {
	buf []byte
	n   byte
}

func newIncStream(blockSize int) *incStream {
	x := new(incStream)
	x.buf = make([]byte, blockSize)
	return x
}

func (x *incStream) Next() []byte {
	x.n++
	for i := range x.buf {
		x.buf[i] = x.n
		x.n++
	}
	return x.buf
}

func testXorWriter(t *testing.T, maxio int) {
	var plain, crypt [256]byte
	for i := 0; i < len(plain); i++ {
		plain[i] = byte(i)
	}
	b := new(bytes.Buffer)
	for block := 1; block <= 64 && block <= maxio; block *= 2 {
		// compute encrypted version
		n := byte(0)
		for i := 0; i < len(crypt); i++ {
			if i%block == 0 {
				n++
			}
			crypt[i] = plain[i] ^ n
			n++
		}

		for frag := 0; frag < 2; frag++ {
			test := fmt.Sprintf("block=%d frag=%d maxio=%d", block, frag, maxio)
			b.Reset()
			r := bytes.NewBuffer(plain[0:])
			s := newIncStream(block)
			w := newXorWriter(s, b)

			// copy plain into w in increasingly large chunks: 1, 1, 2, 4, 8, ...
			// if frag != 0, move the 1 to the end to cause fragmentation.
			if frag == 0 {
				_, err := io.Copyn(w, r, 1)
				if err != nil {
					t.Errorf("%s: first Copyn: %s", test, err)
					continue
				}
			}
			for n := 1; n <= len(plain)/2; n *= 2 {
				_, err := io.Copyn(w, r, int64(n))
				if err != nil {
					t.Errorf("%s: Copyn %d: %s", test, n, err)
				}
			}

			// check output
			crypt := crypt[0 : len(crypt)-frag]
			data := b.Bytes()
			if len(data) != len(crypt) {
				t.Errorf("%s: want %d bytes, got %d", test, len(crypt), len(data))
				continue
			}

			if string(data) != string(crypt) {
				t.Errorf("%s: want %x got %x", test, data, crypt)
			}
		}
	}
}


func TestXorWriter(t *testing.T) {
	// Do shorter I/O sizes first; they're easier to debug.
	for n := 1; n <= 256 && !t.Failed(); n *= 2 {
		testXorWriter(t, n)
	}
}

func testXorReader(t *testing.T, maxio int) {
	var readers = []func(io.Reader) io.Reader{
		func(r io.Reader) io.Reader { return r },
		iotest.OneByteReader,
		iotest.HalfReader,
	}
	var plain, crypt [256]byte
	for i := 0; i < len(plain); i++ {
		plain[i] = byte(255 - i)
	}
	b := new(bytes.Buffer)
	for block := 1; block <= 64 && block <= maxio; block *= 2 {
		// compute encrypted version
		n := byte(0)
		for i := 0; i < len(crypt); i++ {
			if i%block == 0 {
				n++
			}
			crypt[i] = plain[i] ^ n
			n++
		}

		for mode := 0; mode < len(readers); mode++ {
			for frag := 0; frag < 2; frag++ {
				test := fmt.Sprintf("block=%d mode=%d frag=%d maxio=%d", block, mode, frag, maxio)
				s := newIncStream(block)
				b.Reset()
				r := newXorReader(s, readers[mode](bytes.NewBuffer(crypt[0:maxio])))

				// read from crypt in increasingly large chunks: 1, 1, 2, 4, 8, ...
				// if frag == 1, move the 1 to the end to cause fragmentation.
				if frag == 0 {
					_, err := io.Copyn(b, r, 1)
					if err != nil {
						t.Errorf("%s: first Copyn: %s", test, err)
						continue
					}
				}
				for n := 1; n <= maxio/2; n *= 2 {
					_, err := io.Copyn(b, r, int64(n))
					if err != nil {
						t.Errorf("%s: Copyn %d: %s", test, n, err)
					}
				}

				// check output
				data := b.Bytes()
				crypt := crypt[0 : maxio-frag]
				plain := plain[0 : maxio-frag]
				if len(data) != len(plain) {
					t.Errorf("%s: want %d bytes, got %d", test, len(plain), len(data))
					continue
				}

				if string(data) != string(plain) {
					t.Errorf("%s: input=%x want %x got %x", test, crypt, plain, data)
				}
			}
		}
	}
}

func TestXorReader(t *testing.T) {
	// Do shorter I/O sizes first; they're easier to debug.
	for n := 1; n <= 256 && !t.Failed(); n *= 2 {
		testXorReader(t, n)
	}
}

// TODO(rsc): Test handling of writes after write errors.
