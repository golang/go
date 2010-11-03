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

// Simple Cipher for testing: adds an incrementing amount
// to each byte in each
type IncCipher struct {
	blockSize  int
	delta      byte
	encrypting bool
}

func (c *IncCipher) BlockSize() int { return c.blockSize }

func (c *IncCipher) Encrypt(dst, src []byte) {
	if !c.encrypting {
		panic("encrypt: not encrypting")
	}
	if len(src) != c.blockSize || len(dst) != c.blockSize {
		panic(fmt.Sprintln("encrypt: wrong block size", c.blockSize, len(src), len(dst)))
	}
	c.delta++
	for i, b := range src {
		dst[i] = b + c.delta
	}
}

func (c *IncCipher) Decrypt(dst, src []byte) {
	if c.encrypting {
		panic("decrypt: not decrypting")
	}
	if len(src) != c.blockSize || len(dst) != c.blockSize {
		panic(fmt.Sprintln("decrypt: wrong block size ", c.blockSize, " ", len(src), " ", len(dst)))
	}
	c.delta--
	for i, b := range src {
		dst[i] = b + c.delta
	}
}

func TestECBEncrypter(t *testing.T) {
	var plain, crypt [256]byte
	for i := 0; i < len(plain); i++ {
		plain[i] = byte(i)
	}
	b := new(bytes.Buffer)
	for block := 1; block <= 64; block *= 2 {
		// compute encrypted version
		delta := byte(0)
		for i := 0; i < len(crypt); i++ {
			if i%block == 0 {
				delta++
			}
			crypt[i] = plain[i] + delta
		}

		for frag := 0; frag < 2; frag++ {
			c := &IncCipher{block, 0, true}
			b.Reset()
			r := bytes.NewBuffer(plain[0:])
			w := NewECBEncrypter(c, b)

			// copy plain into w in increasingly large chunks: 1, 1, 2, 4, 8, ...
			// if frag != 0, move the 1 to the end to cause fragmentation.
			if frag == 0 {
				_, err := io.Copyn(w, r, 1)
				if err != nil {
					t.Errorf("block=%d frag=0: first Copyn: %s", block, err)
					continue
				}
			}
			for n := 1; n <= len(plain)/2; n *= 2 {
				_, err := io.Copyn(w, r, int64(n))
				if err != nil {
					t.Errorf("block=%d frag=%d: Copyn %d: %s", block, frag, n, err)
				}
			}
			if frag != 0 {
				_, err := io.Copyn(w, r, 1)
				if err != nil {
					t.Errorf("block=%d frag=1: last Copyn: %s", block, err)
					continue
				}
			}

			// check output
			data := b.Bytes()
			if len(data) != len(crypt) {
				t.Errorf("block=%d frag=%d: want %d bytes, got %d", block, frag, len(crypt), len(data))
				continue
			}

			if string(data) != string(crypt[0:]) {
				t.Errorf("block=%d frag=%d: want %x got %x", block, frag, data, crypt)
			}
		}
	}
}

func testECBDecrypter(t *testing.T, maxio int) {
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
		delta := byte(0)
		for i := 0; i < len(crypt); i++ {
			if i%block == 0 {
				delta++
			}
			crypt[i] = plain[i] + delta
		}

		for mode := 0; mode < len(readers); mode++ {
			for frag := 0; frag < 2; frag++ {
				test := fmt.Sprintf("block=%d mode=%d frag=%d maxio=%d", block, mode, frag, maxio)
				c := &IncCipher{block, 0, false}
				b.Reset()
				r := NewECBDecrypter(c, readers[mode](bytes.NewBuffer(crypt[0:maxio])))

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
				if frag != 0 {
					_, err := io.Copyn(b, r, 1)
					if err != nil {
						t.Errorf("%s: last Copyn: %s", test, err)
						continue
					}
				}

				// check output
				data := b.Bytes()
				if len(data) != maxio {
					t.Errorf("%s: want %d bytes, got %d", test, maxio, len(data))
					continue
				}

				if string(data) != string(plain[0:maxio]) {
					t.Errorf("%s: input=%x want %x got %x", test, crypt[0:maxio], plain[0:maxio], data)
				}
			}
		}
	}
}

func TestECBDecrypter(t *testing.T) {
	// Do shorter I/O sizes first; they're easier to debug.
	for n := 1; n <= 256 && !t.Failed(); n *= 2 {
		testECBDecrypter(t, n)
	}
}
