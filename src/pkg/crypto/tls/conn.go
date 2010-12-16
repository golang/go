// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TLS low level connection and record layer

package tls

import (
	"bytes"
	"crypto/cipher"
	"crypto/subtle"
	"crypto/x509"
	"hash"
	"io"
	"net"
	"os"
	"sync"
)

// A Conn represents a secured connection.
// It implements the net.Conn interface.
type Conn struct {
	// constant
	conn     net.Conn
	isClient bool

	// constant after handshake; protected by handshakeMutex
	handshakeMutex    sync.Mutex // handshakeMutex < in.Mutex, out.Mutex, errMutex
	vers              uint16     // TLS version
	haveVers          bool       // version has been negotiated
	config            *Config    // configuration passed to constructor
	handshakeComplete bool
	cipherSuite       uint16
	ocspResponse      []byte // stapled OCSP response
	peerCertificates  []*x509.Certificate

	clientProtocol string

	// first permanent error
	errMutex sync.Mutex
	err      os.Error

	// input/output
	in, out  halfConn     // in.Mutex < out.Mutex
	rawInput *block       // raw input, right off the wire
	input    *block       // application data waiting to be read
	hand     bytes.Buffer // handshake data waiting to be read

	tmp [16]byte
}

func (c *Conn) setError(err os.Error) os.Error {
	c.errMutex.Lock()
	defer c.errMutex.Unlock()

	if c.err == nil {
		c.err = err
	}
	return err
}

func (c *Conn) error() os.Error {
	c.errMutex.Lock()
	defer c.errMutex.Unlock()

	return c.err
}

// Access to net.Conn methods.
// Cannot just embed net.Conn because that would
// export the struct field too.

// LocalAddr returns the local network address.
func (c *Conn) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

// RemoteAddr returns the remote network address.
func (c *Conn) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

// SetTimeout sets the read deadline associated with the connection.
// There is no write deadline.
func (c *Conn) SetTimeout(nsec int64) os.Error {
	return c.conn.SetTimeout(nsec)
}

// SetReadTimeout sets the time (in nanoseconds) that
// Read will wait for data before returning os.EAGAIN.
// Setting nsec == 0 (the default) disables the deadline.
func (c *Conn) SetReadTimeout(nsec int64) os.Error {
	return c.conn.SetReadTimeout(nsec)
}

// SetWriteTimeout exists to satisfy the net.Conn interface
// but is not implemented by TLS.  It always returns an error.
func (c *Conn) SetWriteTimeout(nsec int64) os.Error {
	return os.NewError("TLS does not support SetWriteTimeout")
}

// A halfConn represents one direction of the record layer
// connection, either sending or receiving.
type halfConn struct {
	sync.Mutex
	cipher interface{} // cipher algorithm
	mac    hash.Hash   // MAC algorithm
	seq    [8]byte     // 64-bit sequence number
	bfree  *block      // list of free blocks

	nextCipher interface{} // next encryption state
	nextMac    hash.Hash   // next MAC algorithm
}

// prepareCipherSpec sets the encryption and MAC states
// that a subsequent changeCipherSpec will use.
func (hc *halfConn) prepareCipherSpec(cipher interface{}, mac hash.Hash) {
	hc.nextCipher = cipher
	hc.nextMac = mac
}

// changeCipherSpec changes the encryption and MAC states
// to the ones previously passed to prepareCipherSpec.
func (hc *halfConn) changeCipherSpec() os.Error {
	if hc.nextCipher == nil {
		return alertInternalError
	}
	hc.cipher = hc.nextCipher
	hc.mac = hc.nextMac
	hc.nextCipher = nil
	hc.nextMac = nil
	return nil
}

// incSeq increments the sequence number.
func (hc *halfConn) incSeq() {
	for i := 7; i >= 0; i-- {
		hc.seq[i]++
		if hc.seq[i] != 0 {
			return
		}
	}

	// Not allowed to let sequence number wrap.
	// Instead, must renegotiate before it does.
	// Not likely enough to bother.
	panic("TLS: sequence number wraparound")
}

// resetSeq resets the sequence number to zero.
func (hc *halfConn) resetSeq() {
	for i := range hc.seq {
		hc.seq[i] = 0
	}
}

// removePadding returns an unpadded slice, in constant time, which is a prefix
// of the input. It also returns a byte which is equal to 255 if the padding
// was valid and 0 otherwise. See RFC 2246, section 6.2.3.2
func removePadding(payload []byte) ([]byte, byte) {
	if len(payload) < 1 {
		return payload, 0
	}

	paddingLen := payload[len(payload)-1]
	t := uint(len(payload)-1) - uint(paddingLen)
	// if len(payload) >= (paddingLen - 1) then the MSB of t is zero
	good := byte(int32(^t) >> 31)

	toCheck := 255 // the maximum possible padding length
	// The length of the padded data is public, so we can use an if here
	if toCheck+1 > len(payload) {
		toCheck = len(payload) - 1
	}

	for i := 0; i < toCheck; i++ {
		t := uint(paddingLen) - uint(i)
		// if i <= paddingLen then the MSB of t is zero
		mask := byte(int32(^t) >> 31)
		b := payload[len(payload)-1-i]
		good &^= mask&paddingLen ^ mask&b
	}

	// We AND together the bits of good and replicate the result across
	// all the bits.
	good &= good << 4
	good &= good << 2
	good &= good << 1
	good = uint8(int8(good) >> 7)

	toRemove := good&paddingLen + 1
	return payload[:len(payload)-int(toRemove)], good
}

func roundUp(a, b int) int {
	return a + (b-a%b)%b
}

// decrypt checks and strips the mac and decrypts the data in b.
func (hc *halfConn) decrypt(b *block) (bool, alert) {
	// pull out payload
	payload := b.data[recordHeaderLen:]

	macSize := 0
	if hc.mac != nil {
		macSize = hc.mac.Size()
	}

	paddingGood := byte(255)

	// decrypt
	if hc.cipher != nil {
		switch c := hc.cipher.(type) {
		case cipher.Stream:
			c.XORKeyStream(payload, payload)
		case cipher.BlockMode:
			blockSize := c.BlockSize()

			if len(payload)%blockSize != 0 || len(payload) < roundUp(macSize+1, blockSize) {
				return false, alertBadRecordMAC
			}

			c.CryptBlocks(payload, payload)
			payload, paddingGood = removePadding(payload)
			b.resize(recordHeaderLen + len(payload))

			// note that we still have a timing side-channel in the
			// MAC check, below. An attacker can align the record
			// so that a correct padding will cause one less hash
			// block to be calculated. Then they can iteratively
			// decrypt a record by breaking each byte. See
			// "Password Interception in a SSL/TLS Channel", Brice
			// Canvel et al.
			//
			// However, our behaviour matches OpenSSL, so we leak
			// only as much as they do.
		default:
			panic("unknown cipher type")
		}
	}

	// check, strip mac
	if hc.mac != nil {
		if len(payload) < macSize {
			return false, alertBadRecordMAC
		}

		// strip mac off payload, b.data
		n := len(payload) - macSize
		b.data[3] = byte(n >> 8)
		b.data[4] = byte(n)
		b.resize(recordHeaderLen + n)
		remoteMAC := payload[n:]

		hc.mac.Reset()
		hc.mac.Write(hc.seq[0:])
		hc.incSeq()
		hc.mac.Write(b.data)

		if subtle.ConstantTimeCompare(hc.mac.Sum(), remoteMAC) != 1 || paddingGood != 255 {
			return false, alertBadRecordMAC
		}
	}

	return true, 0
}

// padToBlockSize calculates the needed padding block, if any, for a payload.
// On exit, prefix aliases payload and extends to the end of the last full
// block of payload. finalBlock is a fresh slice which contains the contents of
// any suffix of payload as well as the needed padding to make finalBlock a
// full block.
func padToBlockSize(payload []byte, blockSize int) (prefix, finalBlock []byte) {
	overrun := len(payload) % blockSize
	paddingLen := blockSize - overrun
	prefix = payload[:len(payload)-overrun]
	finalBlock = make([]byte, blockSize)
	copy(finalBlock, payload[len(payload)-overrun:])
	for i := overrun; i < blockSize; i++ {
		finalBlock[i] = byte(paddingLen - 1)
	}
	return
}

// encrypt encrypts and macs the data in b.
func (hc *halfConn) encrypt(b *block) (bool, alert) {
	// mac
	if hc.mac != nil {
		hc.mac.Reset()
		hc.mac.Write(hc.seq[0:])
		hc.incSeq()
		hc.mac.Write(b.data)
		mac := hc.mac.Sum()
		n := len(b.data)
		b.resize(n + len(mac))
		copy(b.data[n:], mac)
	}

	payload := b.data[recordHeaderLen:]

	// encrypt
	if hc.cipher != nil {
		switch c := hc.cipher.(type) {
		case cipher.Stream:
			c.XORKeyStream(payload, payload)
		case cipher.BlockMode:
			prefix, finalBlock := padToBlockSize(payload, c.BlockSize())
			b.resize(recordHeaderLen + len(prefix) + len(finalBlock))
			c.CryptBlocks(b.data[recordHeaderLen:], prefix)
			c.CryptBlocks(b.data[recordHeaderLen+len(prefix):], finalBlock)
		default:
			panic("unknown cipher type")
		}
	}

	// update length to include MAC and any block padding needed.
	n := len(b.data) - recordHeaderLen
	b.data[3] = byte(n >> 8)
	b.data[4] = byte(n)

	return true, 0
}

// A block is a simple data buffer.
type block struct {
	data []byte
	off  int // index for Read
	link *block
}

// resize resizes block to be n bytes, growing if necessary.
func (b *block) resize(n int) {
	if n > cap(b.data) {
		b.reserve(n)
	}
	b.data = b.data[0:n]
}

// reserve makes sure that block contains a capacity of at least n bytes.
func (b *block) reserve(n int) {
	if cap(b.data) >= n {
		return
	}
	m := cap(b.data)
	if m == 0 {
		m = 1024
	}
	for m < n {
		m *= 2
	}
	data := make([]byte, len(b.data), m)
	copy(data, b.data)
	b.data = data
}

// readFromUntil reads from r into b until b contains at least n bytes
// or else returns an error.
func (b *block) readFromUntil(r io.Reader, n int) os.Error {
	// quick case
	if len(b.data) >= n {
		return nil
	}

	// read until have enough.
	b.reserve(n)
	for {
		m, err := r.Read(b.data[len(b.data):cap(b.data)])
		b.data = b.data[0 : len(b.data)+m]
		if len(b.data) >= n {
			break
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (b *block) Read(p []byte) (n int, err os.Error) {
	n = copy(p, b.data[b.off:])
	b.off += n
	return
}

// newBlock allocates a new block, from hc's free list if possible.
func (hc *halfConn) newBlock() *block {
	b := hc.bfree
	if b == nil {
		return new(block)
	}
	hc.bfree = b.link
	b.link = nil
	b.resize(0)
	return b
}

// freeBlock returns a block to hc's free list.
// The protocol is such that each side only has a block or two on
// its free list at a time, so there's no need to worry about
// trimming the list, etc.
func (hc *halfConn) freeBlock(b *block) {
	b.link = hc.bfree
	hc.bfree = b
}

// splitBlock splits a block after the first n bytes,
// returning a block with those n bytes and a
// block with the remaindec.  the latter may be nil.
func (hc *halfConn) splitBlock(b *block, n int) (*block, *block) {
	if len(b.data) <= n {
		return b, nil
	}
	bb := hc.newBlock()
	bb.resize(len(b.data) - n)
	copy(bb.data, b.data[n:])
	b.data = b.data[0:n]
	return b, bb
}

// readRecord reads the next TLS record from the connection
// and updates the record layer state.
// c.in.Mutex <= L; c.input == nil.
func (c *Conn) readRecord(want recordType) os.Error {
	// Caller must be in sync with connection:
	// handshake data if handshake not yet completed,
	// else application data.  (We don't support renegotiation.)
	switch want {
	default:
		return c.sendAlert(alertInternalError)
	case recordTypeHandshake, recordTypeChangeCipherSpec:
		if c.handshakeComplete {
			return c.sendAlert(alertInternalError)
		}
	case recordTypeApplicationData:
		if !c.handshakeComplete {
			return c.sendAlert(alertInternalError)
		}
	}

Again:
	if c.rawInput == nil {
		c.rawInput = c.in.newBlock()
	}
	b := c.rawInput

	// Read header, payload.
	if err := b.readFromUntil(c.conn, recordHeaderLen); err != nil {
		// RFC suggests that EOF without an alertCloseNotify is
		// an error, but popular web sites seem to do this,
		// so we can't make it an error.
		// if err == os.EOF {
		// 	err = io.ErrUnexpectedEOF
		// }
		if e, ok := err.(net.Error); !ok || !e.Temporary() {
			c.setError(err)
		}
		return err
	}
	typ := recordType(b.data[0])
	vers := uint16(b.data[1])<<8 | uint16(b.data[2])
	n := int(b.data[3])<<8 | int(b.data[4])
	if c.haveVers && vers != c.vers {
		return c.sendAlert(alertProtocolVersion)
	}
	if n > maxCiphertext {
		return c.sendAlert(alertRecordOverflow)
	}
	if err := b.readFromUntil(c.conn, recordHeaderLen+n); err != nil {
		if err == os.EOF {
			err = io.ErrUnexpectedEOF
		}
		if e, ok := err.(net.Error); !ok || !e.Temporary() {
			c.setError(err)
		}
		return err
	}

	// Process message.
	b, c.rawInput = c.in.splitBlock(b, recordHeaderLen+n)
	b.off = recordHeaderLen
	if ok, err := c.in.decrypt(b); !ok {
		return c.sendAlert(err)
	}
	data := b.data[b.off:]
	if len(data) > maxPlaintext {
		c.sendAlert(alertRecordOverflow)
		c.in.freeBlock(b)
		return c.error()
	}

	switch typ {
	default:
		c.sendAlert(alertUnexpectedMessage)

	case recordTypeAlert:
		if len(data) != 2 {
			c.sendAlert(alertUnexpectedMessage)
			break
		}
		if alert(data[1]) == alertCloseNotify {
			c.setError(os.EOF)
			break
		}
		switch data[0] {
		case alertLevelWarning:
			// drop on the floor
			c.in.freeBlock(b)
			goto Again
		case alertLevelError:
			c.setError(&net.OpError{Op: "remote error", Error: alert(data[1])})
		default:
			c.sendAlert(alertUnexpectedMessage)
		}

	case recordTypeChangeCipherSpec:
		if typ != want || len(data) != 1 || data[0] != 1 {
			c.sendAlert(alertUnexpectedMessage)
			break
		}
		err := c.in.changeCipherSpec()
		if err != nil {
			c.sendAlert(err.(alert))
		}

	case recordTypeApplicationData:
		if typ != want {
			c.sendAlert(alertUnexpectedMessage)
			break
		}
		c.input = b
		b = nil

	case recordTypeHandshake:
		// TODO(rsc): Should at least pick off connection close.
		if typ != want {
			return c.sendAlert(alertNoRenegotiation)
		}
		c.hand.Write(data)
	}

	if b != nil {
		c.in.freeBlock(b)
	}
	return c.error()
}

// sendAlert sends a TLS alert message.
// c.out.Mutex <= L.
func (c *Conn) sendAlertLocked(err alert) os.Error {
	c.tmp[0] = alertLevelError
	if err == alertNoRenegotiation {
		c.tmp[0] = alertLevelWarning
	}
	c.tmp[1] = byte(err)
	c.writeRecord(recordTypeAlert, c.tmp[0:2])
	// closeNotify is a special case in that it isn't an error:
	if err != alertCloseNotify {
		return c.setError(&net.OpError{Op: "local error", Error: err})
	}
	return nil
}

// sendAlert sends a TLS alert message.
// L < c.out.Mutex.
func (c *Conn) sendAlert(err alert) os.Error {
	c.out.Lock()
	defer c.out.Unlock()
	return c.sendAlertLocked(err)
}

// writeRecord writes a TLS record with the given type and payload
// to the connection and updates the record layer state.
// c.out.Mutex <= L.
func (c *Conn) writeRecord(typ recordType, data []byte) (n int, err os.Error) {
	b := c.out.newBlock()
	for len(data) > 0 {
		m := len(data)
		if m > maxPlaintext {
			m = maxPlaintext
		}
		b.resize(recordHeaderLen + m)
		b.data[0] = byte(typ)
		vers := c.vers
		if vers == 0 {
			vers = maxVersion
		}
		b.data[1] = byte(vers >> 8)
		b.data[2] = byte(vers)
		b.data[3] = byte(m >> 8)
		b.data[4] = byte(m)
		copy(b.data[recordHeaderLen:], data)
		c.out.encrypt(b)
		_, err = c.conn.Write(b.data)
		if err != nil {
			break
		}
		n += m
		data = data[m:]
	}
	c.out.freeBlock(b)

	if typ == recordTypeChangeCipherSpec {
		err = c.out.changeCipherSpec()
		if err != nil {
			// Cannot call sendAlert directly,
			// because we already hold c.out.Mutex.
			c.tmp[0] = alertLevelError
			c.tmp[1] = byte(err.(alert))
			c.writeRecord(recordTypeAlert, c.tmp[0:2])
			c.err = &net.OpError{Op: "local error", Error: err}
			return n, c.err
		}
	}
	return
}

// readHandshake reads the next handshake message from
// the record layer.
// c.in.Mutex < L; c.out.Mutex < L.
func (c *Conn) readHandshake() (interface{}, os.Error) {
	for c.hand.Len() < 4 {
		if c.err != nil {
			return nil, c.err
		}
		c.readRecord(recordTypeHandshake)
	}

	data := c.hand.Bytes()
	n := int(data[1])<<16 | int(data[2])<<8 | int(data[3])
	if n > maxHandshake {
		c.sendAlert(alertInternalError)
		return nil, c.err
	}
	for c.hand.Len() < 4+n {
		if c.err != nil {
			return nil, c.err
		}
		c.readRecord(recordTypeHandshake)
	}
	data = c.hand.Next(4 + n)
	var m handshakeMessage
	switch data[0] {
	case typeClientHello:
		m = new(clientHelloMsg)
	case typeServerHello:
		m = new(serverHelloMsg)
	case typeCertificate:
		m = new(certificateMsg)
	case typeCertificateRequest:
		m = new(certificateRequestMsg)
	case typeCertificateStatus:
		m = new(certificateStatusMsg)
	case typeServerKeyExchange:
		m = new(serverKeyExchangeMsg)
	case typeServerHelloDone:
		m = new(serverHelloDoneMsg)
	case typeClientKeyExchange:
		m = new(clientKeyExchangeMsg)
	case typeCertificateVerify:
		m = new(certificateVerifyMsg)
	case typeNextProtocol:
		m = new(nextProtoMsg)
	case typeFinished:
		m = new(finishedMsg)
	default:
		c.sendAlert(alertUnexpectedMessage)
		return nil, alertUnexpectedMessage
	}

	// The handshake message unmarshallers
	// expect to be able to keep references to data,
	// so pass in a fresh copy that won't be overwritten.
	data = append([]byte(nil), data...)

	if !m.unmarshal(data) {
		c.sendAlert(alertUnexpectedMessage)
		return nil, alertUnexpectedMessage
	}
	return m, nil
}

// Write writes data to the connection.
func (c *Conn) Write(b []byte) (n int, err os.Error) {
	if err = c.Handshake(); err != nil {
		return
	}

	c.out.Lock()
	defer c.out.Unlock()

	if !c.handshakeComplete {
		return 0, alertInternalError
	}
	if c.err != nil {
		return 0, c.err
	}
	return c.writeRecord(recordTypeApplicationData, b)
}

// Read can be made to time out and return err == os.EAGAIN
// after a fixed time limit; see SetTimeout and SetReadTimeout.
func (c *Conn) Read(b []byte) (n int, err os.Error) {
	if err = c.Handshake(); err != nil {
		return
	}

	c.in.Lock()
	defer c.in.Unlock()

	for c.input == nil && c.err == nil {
		if err := c.readRecord(recordTypeApplicationData); err != nil {
			// Soft error, like EAGAIN
			return 0, err
		}
	}
	if c.err != nil {
		return 0, c.err
	}
	n, err = c.input.Read(b)
	if c.input.off >= len(c.input.data) {
		c.in.freeBlock(c.input)
		c.input = nil
	}
	return n, nil
}

// Close closes the connection.
func (c *Conn) Close() os.Error {
	if err := c.Handshake(); err != nil {
		return err
	}
	return c.sendAlert(alertCloseNotify)
}

// Handshake runs the client or server handshake
// protocol if it has not yet been run.
// Most uses of this package need not call Handshake
// explicitly: the first Read or Write will call it automatically.
func (c *Conn) Handshake() os.Error {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()
	if err := c.error(); err != nil {
		return err
	}
	if c.handshakeComplete {
		return nil
	}
	if c.isClient {
		return c.clientHandshake()
	}
	return c.serverHandshake()
}

// ConnectionState returns basic TLS details about the connection.
func (c *Conn) ConnectionState() ConnectionState {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()

	var state ConnectionState
	state.HandshakeComplete = c.handshakeComplete
	if c.handshakeComplete {
		state.NegotiatedProtocol = c.clientProtocol
		state.CipherSuite = c.cipherSuite
	}

	return state
}

// OCSPResponse returns the stapled OCSP response from the TLS server, if
// any. (Only valid for client connections.)
func (c *Conn) OCSPResponse() []byte {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()

	return c.ocspResponse
}

// PeerCertificates returns the certificate chain that was presented by the
// other side.
func (c *Conn) PeerCertificates() []*x509.Certificate {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()

	return c.peerCertificates
}

// VerifyHostname checks that the peer certificate chain is valid for
// connecting to host.  If so, it returns nil; if not, it returns an os.Error
// describing the problem.
func (c *Conn) VerifyHostname(host string) os.Error {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()
	if !c.isClient {
		return os.ErrorString("VerifyHostname called on TLS server connection")
	}
	if !c.handshakeComplete {
		return os.ErrorString("TLS handshake has not yet been performed")
	}
	return c.peerCertificates[0].VerifyHostname(host)
}
