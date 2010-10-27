// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Electronic codebook (ECB) mode.
// ECB is a fancy name for ``encrypt and decrypt each block separately.''
// It's a pretty bad thing to do for any large amount of data (more than one block),
// because the individual blocks can still be identified, duplicated, and reordered.
// The ECB implementation exists mainly to provide buffering for
// the other modes, which wrap it by providing modified Ciphers.

// See NIST SP 800-38A, pp 9-10

package block

import (
	"io"
	"os"
	"strconv"
)

type ecbDecrypter struct {
	c         Cipher
	r         io.Reader
	blockSize int // block size

	// Buffered data.
	// The buffer buf is used as storage for both
	// plain or crypt; at least one of those is nil at any given time.
	buf   []byte
	plain []byte // plain text waiting to be read
	crypt []byte // ciphertext waiting to be decrypted
}

// Read into x.crypt until it has a full block or EOF or an error happens.
func (x *ecbDecrypter) fillCrypt() os.Error {
	var err os.Error
	for len(x.crypt) < x.blockSize {
		off := len(x.crypt)
		var m int
		m, err = x.r.Read(x.crypt[off:x.blockSize])
		x.crypt = x.crypt[0 : off+m]
		if m == 0 {
			break
		}

		// If an error happened but we got enough
		// data to do some decryption, we can decrypt
		// first and report the error (with some data) later.
		// But if we don't have enough to decrypt,
		// have to stop now.
		if err != nil && len(x.crypt) < x.blockSize {
			break
		}
	}
	return err
}

// Read from plain text buffer into p.
func (x *ecbDecrypter) readPlain(p []byte) int {
	n := len(x.plain)
	if n > len(p) {
		n = len(p)
	}
	for i := 0; i < n; i++ {
		p[i] = x.plain[i]
	}
	if n < len(x.plain) {
		x.plain = x.plain[n:]
	} else {
		x.plain = nil
	}
	return n
}

type ecbFragmentError int

func (n ecbFragmentError) String() string {
	return "crypto/block: " + strconv.Itoa(int(n)) + "-byte fragment at EOF"
}

func (x *ecbDecrypter) Read(p []byte) (n int, err os.Error) {
	if len(p) == 0 {
		return
	}

	// If there's no plaintext waiting and p is not big enough
	// to hold a whole cipher block, we'll have to work in the
	// cipher text buffer.  Set it to non-nil so that the
	// code below will fill it.
	if x.plain == nil && len(p) < x.blockSize && x.crypt == nil {
		x.crypt = x.buf[0:0]
	}

	// If there is a leftover cipher text buffer,
	// try to accumulate a full block.
	if x.crypt != nil {
		err = x.fillCrypt()
		if err != nil || len(x.crypt) == 0 {
			return
		}
		x.c.Decrypt(x.crypt, x.crypt)
		x.plain = x.crypt
		x.crypt = nil
	}

	// If there is a leftover plain text buffer, read from it.
	if x.plain != nil {
		n = x.readPlain(p)
		return
	}

	// Read and decrypt directly in caller's buffer.
	n, err = io.ReadAtLeast(x.r, p, x.blockSize)
	if err == os.EOF && n > 0 {
		// EOF is only okay on block boundary
		err = os.ErrorString("block fragment at EOF during decryption")
		return
	}
	var i int
	for i = 0; i+x.blockSize <= n; i += x.blockSize {
		a := p[i : i+x.blockSize]
		x.c.Decrypt(a, a)
	}

	// There might be an encrypted fringe remaining.
	// Save it for next time.
	if i < n {
		p = p[i:n]
		copy(x.buf, p)
		x.crypt = x.buf[0:len(p)]
		n = i
	}

	return
}

// NewECBDecrypter returns a reader that reads data from r and decrypts it using c.
// It decrypts by calling c.Decrypt on each block in sequence;
// this mode is known as electronic codebook mode, or ECB.
// The returned Reader does not buffer or read ahead except
// as required by the cipher's block size.
func NewECBDecrypter(c Cipher, r io.Reader) io.Reader {
	x := new(ecbDecrypter)
	x.c = c
	x.r = r
	x.blockSize = c.BlockSize()
	x.buf = make([]byte, x.blockSize)
	return x
}

type ecbEncrypter struct {
	c         Cipher
	w         io.Writer
	blockSize int

	// Buffered data.
	// The buffer buf is used as storage for both
	// plain or crypt.  If both are non-nil, plain
	// follows crypt in buf.
	buf   []byte
	plain []byte // plain text waiting to be encrypted
	crypt []byte // encrypted text waiting to be written
}

// Flush the x.crypt buffer to x.w.
func (x *ecbEncrypter) flushCrypt() os.Error {
	if len(x.crypt) == 0 {
		return nil
	}
	n, err := x.w.Write(x.crypt)
	if n < len(x.crypt) {
		x.crypt = x.crypt[n:]
		if err == nil {
			err = io.ErrShortWrite
		}
	}
	if err != nil {
		return err
	}
	x.crypt = nil
	return nil
}

// Slide x.plain down to the beginning of x.buf.
// Plain is known to have less than one block of data,
// so this is cheap enough.
func (x *ecbEncrypter) slidePlain() {
	if len(x.plain) == 0 {
		x.plain = x.buf[0:0]
	} else if cap(x.plain) < cap(x.buf) {
		copy(x.buf, x.plain)
		x.plain = x.buf[0:len(x.plain)]
	}
}

// Fill x.plain from the data in p.
// Return the number of bytes copied.
func (x *ecbEncrypter) fillPlain(p []byte) int {
	off := len(x.plain)
	n := len(p)
	if max := cap(x.plain) - off; n > max {
		n = max
	}
	x.plain = x.plain[0 : off+n]
	for i := 0; i < n; i++ {
		x.plain[off+i] = p[i]
	}
	return n
}

// Encrypt x.plain; record encrypted range as x.crypt.
func (x *ecbEncrypter) encrypt() {
	var i int
	n := len(x.plain)
	for i = 0; i+x.blockSize <= n; i += x.blockSize {
		a := x.plain[i : i+x.blockSize]
		x.c.Encrypt(a, a)
	}
	x.crypt = x.plain[0:i]
	x.plain = x.plain[i:n]
}

func (x *ecbEncrypter) Write(p []byte) (n int, err os.Error) {
	for {
		// If there is data waiting to be written, write it.
		// This can happen on the first iteration
		// if a write failed in an earlier call.
		if err = x.flushCrypt(); err != nil {
			return
		}

		// Now that encrypted data is gone (flush ran),
		// perhaps we need to slide the plaintext down.
		x.slidePlain()

		// Fill plaintext buffer from p.
		m := x.fillPlain(p)
		if m == 0 {
			break
		}
		n += m
		p = p[m:]

		// Encrypt, adjusting crypt and plain.
		x.encrypt()

		// Write x.crypt.
		if err = x.flushCrypt(); err != nil {
			break
		}
	}
	return
}

// NewECBEncrypter returns a writer that encrypts data using c and writes it to w.
// It encrypts by calling c.Encrypt on each block in sequence;
// this mode is known as electronic codebook mode, or ECB.
// The returned Writer does no buffering except as required
// by the cipher's block size, so there is no need for a Flush method.
func NewECBEncrypter(c Cipher, w io.Writer) io.Writer {
	x := new(ecbEncrypter)
	x.c = c
	x.w = w
	x.blockSize = c.BlockSize()

	// Create a buffer that is an integral number of blocks.
	x.buf = make([]byte, 8192/x.blockSize*x.blockSize)
	return x
}
