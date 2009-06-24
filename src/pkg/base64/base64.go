// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package base64 implements base64 encoding as specified by RFC 4648.
package base64

import (
	"io";
	"os";
	"strconv";
)

/*
 * Encodings
 */

// Encoding is a radix 64 encoding/decoding scheme, defined by a
// 64-character alphabet.  The most common encoding is the "base64"
// encoding defined in RFC 4648 and used in MIME (RFC 2045) and PEM
// (RFC 1421).  RFC 4648 also defines an alternate encoding, which is
// the standard encoding with - and _ substituted for + and /.
type Encoding struct {
	encode string;
	decodeMap [256]byte;
}

const encodeStd = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
const encodeURL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

// NewEncoding returns a new Encoding defined by the given alphabet,
// which must be a 64-byte string.
func NewEncoding(encoder string) *Encoding {
	e := new(Encoding);
	e.encode = encoder;
	for i := 0; i < len(e.decodeMap); i++ {
		e.decodeMap[i] = 0xFF;
	}
	for i := 0; i < len(encoder); i++ {
		e.decodeMap[encoder[i]] = byte(i);
	}
	return e;
}

// StdEncoding is the standard base64 encoding, as defined in
// RFC 4648.
var StdEncoding = NewEncoding(encodeStd);

// URLEncoding is the alternate base64 encoding defined in RFC 4648.
// It is typically used in URLs and file names.
var URLEncoding = NewEncoding(encodeURL);

/*
 * Encoder
 */

// Encode encodes src using the encoding enc, writing
// EncodedLen(len(input)) bytes to dst.
// 
// The encoding pads the output to a multiple of 4 bytes,
// so Encode is not appropriate for use on individual blocks
// of a large data stream.  Use NewEncoder() instead.
func (enc *Encoding) Encode(src, dst []byte) {
	if len(src) == 0 {
		return;
	}

	for len(src) > 0 {
		// Unpack 4x 6-bit source blocks into a 4 byte
		// destination quantum
		switch len(src) {
		default:
			dst[3] |= src[2]&0x3F;
			dst[2] |= src[2]>>6;
			fallthrough;
		case 2:
			dst[2] |= (src[1]<<2)&0x3F;
			dst[1] |= src[1]>>4;
			fallthrough;
		case 1:
			dst[1] |= (src[0]<<4)&0x3F;
			dst[0] |= src[0]>>2;
		}

		// Encode 6-bit blocks using the base64 alphabet
		for j := 0; j < 4; j++ {
			dst[j] = enc.encode[dst[j]];
		}

		// Pad the final quantum
		if len(src) < 3 {
			dst[3] = '=';
			if len(src) < 2 {
				dst[2] = '=';
			}
			break;
		}

		src = src[3:len(src)];
		dst = dst[4:len(dst)];
	}
}

// encodeBlocker is a restricted FIFO for byte data that always
// returns byte arrays whose lengths are some multiple of 3.
type encodeBlocker struct {
	// The overflow buffer contains data that should be returned
	// before any data in nextbuf.
	buffer [3]byte;
	bufpos int;
	nextbuf []byte;
}

// put appends the data contained in buf to the encode blocker's
// buffer.  In general, you have to get everything out before you can
// put another array.
func (eb *encodeBlocker) put(buf []byte) {
	if eb.nextbuf != nil {
		panic("there is already a nextbuf");
	}

	// If we have anything in the overflow buffer, fill it up the
	// rest of the way so we can return the overflow buffer.
	bpos := 0;
	if eb.bufpos != 0 {
		for ; eb.bufpos < 3 && bpos < len(buf); eb.bufpos++ {
			eb.buffer[eb.bufpos] = buf[bpos];
			bpos++;
		}
	}

	if bpos < len(buf) {
		eb.nextbuf = buf[bpos:len(buf)];
	}
}

// get retrieves an input quantum aligned byte array from the encode
// blocker.
func (eb *encodeBlocker) get() []byte {
	// If there is data in the overflow buffer, return it first
	if eb.bufpos > 0 {
		if eb.bufpos < 3 {
			// We don't have a full quantum
			return nil;
		}
		eb.bufpos = 0;
		return &eb.buffer;
	}

	// No overflow buffer, so return nextbuf.  However, it has to
	// be quantum-aligned, so copy the tail of the data into the
	// overflow buffer for next time.
	end := len(eb.nextbuf)/3*3;
	for i := end; i < len(eb.nextbuf); i++ {
		eb.buffer[eb.bufpos] = eb.nextbuf[i];
		eb.bufpos++;
	}
	b := eb.nextbuf[0:end];
	eb.nextbuf = nil;
	if end == 0 {
		return nil;
	}
	return b;
}

// size returns the number of bytes remaining in the encode blocker's
// buffer.
func (eb *encodeBlocker) size() int {
	return (eb.bufpos + len(eb.nextbuf))/3*3;
}

type encoder struct {
	w io.Writer;
	enc *Encoding;
	err os.Error;
	eb encodeBlocker;
}

func (e *encoder) Write(b []byte) (int, os.Error) {
	if e.err != nil {
		return 0, e.err;
	}

	e.eb.put(b);

	output := make([]byte, e.eb.size()/3*4);
	opos := 0;

	for {
		block := e.eb.get();
		if block == nil {
			break;
		}
		e.enc.Encode(block, output[opos:len(output)]);
		opos += len(block)/3*4;
	}

	n, err := e.w.Write(output);
	if err != nil {
		e.err = io.ErrShortWrite;
		return n/4*3, e.err;
	}
	return len(b), nil;
}

// Close flushes any pending output from the encoder.  It is an error
// to call Write after calling Close.
func (e *encoder) Close() os.Error {
	// If there's anything left in the buffer, flush it out
	if e.err == nil && e.eb.bufpos > 0 {
		var output [4]byte;
		e.enc.Encode(e.eb.buffer[0:e.eb.bufpos], &output);
		e.eb.bufpos = 0;
		n, err := e.w.Write(&output);
		if err != nil {
			e.err = io.ErrShortWrite;
		}
	}
	return e.err;
}

// NewEncoder returns a new base64 stream encoder.  Data written to
// the returned writer will be encoded using enc and then written to w.
// Base64 encodings operate in 4-byte blocks; when finished
// writing, the caller must Close the returned encoder to flush any
// partially written blocks.
func NewEncoder(enc *Encoding, w io.Writer) io.WriteCloser {
	return &encoder{w: w, enc: enc};
}

// EncodedLen returns the length in bytes of the base64 encoding
// of an input buffer of length n.
func (enc *Encoding) EncodedLen(n int) int {
	return (n+2)/3*4;
}

/*
 * Decoder
 */

type CorruptInputError int64;

func (e CorruptInputError) String() string {
	return "illegal base64 data at input byte" + strconv.Itoa64(int64(e));
}

// decode is like Decode, but returns an additional 'end' value, which
// indicates if end-of-message padding was encountered and thus any
// additional data is an error.  decode also assumes len(src)%4==0,
// since it is meant for internal use.
func (enc *Encoding) decode(src, dst []byte) (n int, end bool, err os.Error) {
	for i := 0; i < len(src)/4 && !end; i++ {
		// Decode quantum using the base64 alphabet
		var dbuf [4]byte;
		dlen := 4;

dbufloop:
		for j := 0; j < 4; j++ {
			in := src[i*4+j];
			if in == '=' && j >= 2 && i == len(src)/4 - 1 {
				// We've reached the end and there's
				// padding
				if src[i*4+3] != '=' {
					return n, false, CorruptInputError(i*4+2);
				}
				dlen = j;
				end = true;
				break dbufloop;
			}
			dbuf[j] = enc.decodeMap[in];
			if dbuf[j] == 0xFF {
				return n, false, CorruptInputError(i*4+j);
			}
		}

		// Pack 4x 6-bit source blocks into 3 byte destination
		// quantum
		switch dlen {
		case 4:
			dst[i*3+2] = dbuf[2]<<6 | dbuf[3];
			fallthrough;
		case 3:
			dst[i*3+1] = dbuf[1]<<4 | dbuf[2]>>2;
			fallthrough;
		case 2:
			dst[i*3+0] = dbuf[0]<<2 | dbuf[1]>>4;
		}
		n += dlen - 1;
	}

	return n, end, nil;
}

// Decode decodes src using the encoding enc.  It writes at most
// DecodedLen(len(src)) bytes to dst and returns the number of bytes
// written.  If src contains invalid base64 data, it will return the
// number of bytes successfully written and CorruptInputError.
func (enc *Encoding) Decode(src, dst []byte) (n int, err os.Error) {
	if len(src)%4 != 0 {
		return 0, CorruptInputError(len(src)/4*4);
	}

	var _ bool;
	n, _, err = enc.decode(src, dst);
	return;
}

// quantumReader wraps a regular reader and ensures that each read
// will return a slice whose length is a multiple of 4-bytes.
type quantumReader struct {
	r io.Reader;
	buf [4]byte;
	buflen int;
}

func (q *quantumReader) Read(p []byte) (int, os.Error) {
	// Copy buffered data into the output
	for i := 0; i < q.buflen; i++ {
		p[i] = q.buf[i];
	}

	// Read more data into the output
	n, err := q.r.Read(p[q.buflen:len(p)]);

	// Buffer tail data that does not fit into the quanta
	end := (q.buflen+n)/4*4;
	for i := end; i < q.buflen+n; i++ {
		q.buf[i-end] = p[i];
	}

	// Is EOF misaligned?
	if err == os.EOF && q.buflen > 0 {
		err = io.ErrUnexpectedEOF;
	}

	return end, err;
}

// decodeBlocker takes a sequence of arbitrary size output byte slices
// and makes them available as a stream of byte slices whose lengths
// are always a multiple of 3.
type decodeBlocker struct {
	output []byte;
	noutput int;
	overflow [3]byte;
	overflowstart int;
}

// flush flushes as much data from the overflow buffer as possible in
// to the current output buffer, reseting the output buffer to nil if
// it fills it up.  It returns the number of bytes written to the
// output buffer.
func (db *decodeBlocker) flush() int {
	// Copy overflow into the beginning of this buffer
	i := 0;
	for ; i < len(db.output) && db.overflowstart < 3; i++ {
		db.output[i] = db.overflow[db.overflowstart];
		db.overflowstart++;
	}
	if i == len(db.output) {
		db.output = nil;
	} else {
		db.output = db.output[i:len(db.output)];
	}
	return i;
}

// use begins using a new output buffer.  Any data that did not fit in
// the previous output buffer will be placed at the beginning of this
// buffer.
func (db *decodeBlocker) use(buf []byte) {
	db.output = buf;
	db.noutput = 0;
	// Copy left-over overflow from the previous buffer into this
	// buffer
	db.noutput += db.flush();
}

// checkout retrieve the next slice to fill with data.  The length of
// the returned slice will always be a multiple of 3.  It returns nil
// if there is no more buffer space.
func (db *decodeBlocker) checkout() []byte {
	// If we can use the output buffer, do so
	if len(db.output) >= 3 {
		end := len(db.output)/3*3;
		return db.output[0:end];
	} else if db.overflowstart == 3 {
		// Fill the overflow buffer
		db.overflowstart = 0;
		return &db.overflow;
	}
	// We're out of space
	return nil;
}

// checking indicates that we're done with the checked-out slice and
// that we wrote count bytes to it.
func (db *decodeBlocker) checkin(count int) {
	if db.overflowstart == 3 {
		// Wrote to the output buffer
		db.noutput += count;
		db.output = db.output[count:len(db.output)];
	} else {
		// Wrote to the overflow buffer.  Flush what we can to
		// the output buffer.
		n := db.flush();
		if n > count {
			n = count;
		}
		db.noutput += n;
	}
}

// remaining returns the number of bytes remaining in the decode
// blocker's buffer.  This will always be a multiple of 3.
func (db *decodeBlocker) remaining() int {
	return (len(db.output)+2)/3*3;
}

// outlen returns the number of bytes written to the output buffer.
func (db *decodeBlocker) outlen() int {
	return db.noutput;
}

type decoder struct {
	r quantumReader;
	enc *Encoding;
	db decodeBlocker;
	err os.Error;
	// Have we definitely reached the end of the message?
	end bool;
}

func min(a int, b int) int {
	if a < b {
		return a;
	}
	return b;
}

func (d *decoder) Read(output []byte) (int, os.Error) {
	if d.err != nil {
		return 0, d.err;
	}

	d.db.use(output);

	var inbuf [512]byte;

	// Read enough data to fill either our input buffer or our
	// output buffer.
	maxin := min(d.db.remaining()/3*4, len(inbuf));
	n, err := d.r.Read(inbuf[0:maxin]);

	// Decode into output buffer.
	ipos := 0;
	for ipos < n {
		outbuf := d.db.checkout();
		if outbuf == nil {
			// Out of output buffer space
			break;
		}

		inlen := min(len(outbuf)/3*4, n - ipos);
		if d.end {
			// We've seen end-of-message padding, but
			// there's more data.  The RFC says this is an
			// error.
			// XXX Should shift character count
			d.err = CorruptInputError(0);
			break;
		}
		count := 0;
		count, d.end, d.err = d.enc.decode(inbuf[ipos:ipos+inlen], outbuf);
		d.db.checkin(count);
		if d.err != nil {
			// XXX Should shift character count
			break;
		}
		ipos += inlen;
	}

	if err != nil && d.err == nil {
		d.err = err;
	}

	return d.db.outlen(), d.err;
}

// NewDecoder constructs a new base64 stream decoder.
func NewDecoder(enc *Encoding, r io.Reader) io.Reader {
	return &decoder{r: quantumReader{r:r},
			enc: enc,
			db: decodeBlocker{overflowstart: 3}};
}

// DecodeLen returns the maximum length in bytes of the decoded data
// corresponding to n bytes of base64-encoded data.
func (enc *Encoding) DecodedLen(n int) int {
	return n/4*3;
}
