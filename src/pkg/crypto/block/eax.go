// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// EAX mode, not a NIST standard (yet).
// EAX provides encryption and authentication.
// EAX targets the same uses as NIST's CCM mode,
// but EAX adds the ability to run in streaming mode.

// See
// http://csrc.nist.gov/groups/ST/toolkit/BCM/documents/proposedmodes/eax/eax-spec.pdf
// http://www.cs.ucdavis.edu/~rogaway/papers/eax.pdf
// What those papers call OMAC is now called CMAC.

package block

import (
	"fmt"
	"hash"
	"io"
	"os"
)

// An EAXTagError is returned when the message has failed to authenticate,
// because the tag at the end of the message stream (Read) does not match
// the tag computed from the message itself (Computed).
type EAXTagError struct {
	Read     []byte
	Computed []byte
}

func (e *EAXTagError) String() string {
	return fmt.Sprintf("crypto/block: EAX tag mismatch: read %x but computed %x", e.Read, e.Computed)
}

func setupEAX(c Cipher, iv, hdr []byte, tagBytes int) (ctrIV, tag []byte, cmac hash.Hash) {
	n := len(iv)
	if n != c.BlockSize() {
		panic(fmt.Sprintln("crypto/block: EAX: iv length", n, "!=", c.BlockSize()))
	}
	buf := make([]byte, n) // zeroed

	// tag = CMAC(0 + iv) ^ CMAC(1 + hdr) ^ CMAC(2 + data)
	cmac = NewCMAC(c)
	cmac.Write(buf) // 0
	cmac.Write(iv)
	sum := cmac.Sum()
	ctrIV = dup(sum)
	tag = dup(sum[0:tagBytes])

	cmac.Reset()
	buf[n-1] = 1
	cmac.Write(buf) // 1
	cmac.Write(hdr)
	sum = cmac.Sum()
	for i := 0; i < tagBytes; i++ {
		tag[i] ^= sum[i]
	}

	cmac.Reset()
	buf[n-1] = 2 // 2
	cmac.Write(buf)

	return
}

func finishEAX(tag []byte, cmac hash.Hash) {
	// Finish CMAC #2 and xor into tag.
	sum := cmac.Sum()
	for i := range tag {
		tag[i] ^= sum[i]
	}
}

// Writer adapter.  Tees writes into both w and cmac.
// Knows that cmac never returns write errors.
type cmacWriter struct {
	w    io.Writer
	cmac hash.Hash
}

func (cw *cmacWriter) Write(p []byte) (n int, err os.Error) {
	n, err = cw.w.Write(p)
	cw.cmac.Write(p[0:n])
	return
}

// An eaxEncrypter implements the EAX encryption mode.
type eaxEncrypter struct {
	ctr io.Writer  // CTR encrypter
	cw  cmacWriter // CTR's output stream
	tag []byte
}

// NewEAXEncrypter creates and returns a new EAX encrypter
// using the given cipher c, initialization vector iv, associated data hdr,
// and tag length tagBytes.  The encrypter's Write method encrypts
// the data it receives and writes that data to w.
// The encrypter's Close method writes a final authenticating tag to w.
func NewEAXEncrypter(c Cipher, iv []byte, hdr []byte, tagBytes int, w io.Writer) io.WriteCloser {
	x := new(eaxEncrypter)

	// Create new CTR instance writing to both
	// w for encrypted output and cmac for digesting.
	x.cw.w = w
	var ctrIV []byte
	ctrIV, x.tag, x.cw.cmac = setupEAX(c, iv, hdr, tagBytes)
	x.ctr = NewCTRWriter(c, ctrIV, &x.cw)
	return x
}

func (x *eaxEncrypter) Write(p []byte) (n int, err os.Error) {
	return x.ctr.Write(p)
}

func (x *eaxEncrypter) Close() os.Error {
	x.ctr = nil // crash if Write is called again

	// Write tag.
	finishEAX(x.tag, x.cw.cmac)
	n, err := x.cw.w.Write(x.tag)
	if n != len(x.tag) && err == nil {
		err = io.ErrShortWrite
	}

	return err
}

// Reader adapter.  Returns data read from r but hangs
// on to the last len(tag) bytes for itself (returns EOF len(tag)
// bytes early).  Also tees all data returned from Read into
// the cmac digest.  The "don't return the last t bytes"
// and the "tee into digest" functionality could be separated,
// but the latter half is trivial.
type cmacReader struct {
	r    io.Reader
	cmac hash.Hash
	tag  []byte
	tmp  []byte
}

func (cr *cmacReader) Read(p []byte) (n int, err os.Error) {
	// TODO(rsc): Maybe fall back to simpler code if
	// we recognize the underlying r as a ByteBuffer
	// or ByteReader.  Then we can just take the last piece
	// off at the start.

	// First, read a tag-sized chunk.
	// It's probably not the tag (unless there's no data).
	tag := cr.tag
	if len(tag) < cap(tag) {
		nt := len(tag)
		nn, err1 := io.ReadFull(cr.r, tag[nt:cap(tag)])
		tag = tag[0 : nt+nn]
		cr.tag = tag
		if err1 != nil {
			return 0, err1
		}
	}

	tagBytes := len(tag)
	if len(p) > 4*tagBytes {
		// If p is big, try to read directly into p to avoid a copy.
		n, err = cr.r.Read(p[tagBytes:])
		if n == 0 {
			goto out
		}
		// copy old tag into p
		for i := 0; i < tagBytes; i++ {
			p[i] = tag[i]
		}
		// copy new tag out of p
		for i := 0; i < tagBytes; i++ {
			tag[i] = p[n+i]
		}
		goto out
	}

	// Otherwise, read into p and then slide data
	n, err = cr.r.Read(p)
	if n == 0 {
		goto out
	}

	// copy tag+p into p+tmp and then swap tmp, tag
	tmp := cr.tmp
	for i := n + tagBytes - 1; i >= 0; i-- {
		var c byte
		if i < tagBytes {
			c = tag[i]
		} else {
			c = p[i-tagBytes]
		}
		if i < n {
			p[i] = c
		} else {
			tmp[i] = c
		}
	}
	cr.tmp, cr.tag = tag, tmp

out:
	cr.cmac.Write(p[0:n])
	return
}

type eaxDecrypter struct {
	ctr io.Reader
	cr  cmacReader
	tag []byte
}

// NewEAXDecrypter creates and returns a new EAX decrypter
// using the given cipher c, initialization vector iv, associated data hdr,
// and tag length tagBytes.  The encrypter's Read method decrypts and
// returns data read from r.  At r's EOF, the encrypter checks the final
// authenticating tag and returns an EAXTagError if the tag is invalid.
// In that case, the message should be discarded.
// Note that the data stream returned from Read cannot be
// assumed to be valid, authenticated data until Read returns
// 0, nil to signal the end of the data.
func NewEAXDecrypter(c Cipher, iv []byte, hdr []byte, tagBytes int, r io.Reader) io.Reader {
	x := new(eaxDecrypter)

	x.cr.r = r
	x.cr.tag = make([]byte, 0, tagBytes)
	x.cr.tmp = make([]byte, 0, tagBytes)
	var ctrIV []byte
	ctrIV, x.tag, x.cr.cmac = setupEAX(c, iv, hdr, tagBytes)
	x.ctr = NewCTRReader(c, ctrIV, &x.cr)
	return x
}

func (x *eaxDecrypter) checkTag() os.Error {
	x.ctr = nil // crash if Read is called again

	finishEAX(x.tag, x.cr.cmac)
	if !same(x.tag, x.cr.tag) {
		e := new(EAXTagError)
		e.Computed = dup(x.tag)
		e.Read = dup(x.cr.tag)
		return e
	}
	return nil
}

func (x *eaxDecrypter) Read(p []byte) (n int, err os.Error) {
	n, err = x.ctr.Read(p)
	if n == 0 && err == nil {
		err = x.checkTag()
	}
	return n, err
}
