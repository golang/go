// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TLS low level connection and record layer

package tls

import (
	"bytes"
	"context"
	"crypto/cipher"
	"crypto/subtle"
	"crypto/x509"
	"errors"
	"fmt"
	"hash"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// A Conn represents a secured, serializable connection.
// It implements the net.Conn interface.
type Conn struct {
	// constant
	conn        net.Conn
	IsClient    bool                        `json:"is_client"`
	handshakeFn func(context.Context) error // (*conn).clientHandshake or serverHandshake

	// HandshakeStatus is 1 if the connection is currently transferring
	// application Data (i.e. is not currently processing a handshake).
	// This field is only to be accessed with sync/atomic.
	HandshakeStatus        uint32 `json:"handshake_status"`
	HandshakePrimaryStatus uint32 `json:"handshake_primary_status"`
	// constant after handshake; protected by handshakeMutex
	handshakeMutex sync.Mutex
	handshakeErr   error   // error resulting from handshake
	Vers           uint16 `json:"Vers"` // TLS Version
	HaveVers       bool    `json:"have_vers"` // Version has been negotiated
	config         *Config // configuration passed to constructor
	// Handshakes counts the number of Handshakes performed on the
	// connection so far. If renegotiation is disabled then this is either
	// zero or one.
	Handshakes       int `json:"handshakes"`
	DidResume        bool `json:"did_resume"` // whether this connection was a session resumption
	CipherSuite      uint16 `json:"cipher_suite"`
	OcspResponse     []byte  `json:"ocsp_response"` // stapled OCSP response
	Scts             [][]byte `json:"Scts"` // signed certificate timestamps from server
	// Todo - need to save these and restore manually
	peerCertificates []*x509.Certificate
	// verifiedChains contains the certificate chains that we built, as
	// opposed to the ones presented by the server.
	// Todo - need to save these and restore manually
	verifiedChains [][]*x509.Certificate
	// ServerName contains the server name indicated by the client, if any.
	ServerName string `json:"server_name"`
	// SecureRenegotiation is true if the server echoed the secure
	// renegotiation extension. (This is meaningless as a server because
	// renegotiation is not supported In that case.)
	SecureRenegotiation bool `json:"secure_renegotiation"`
	// ekm is a closure for exporting keying material.
	ekm func(label string, context []byte, length int) ([]byte, error)
	// ResumptionSecret is the resumption_master_secret for handling
	// NewSessionTicket messages. nil if config.SessionTicketsDisabled.
	ResumptionSecret []byte `json:"resumption_secret"`

	// ticketKeys is the set of active session ticket keys for this
	// connection. The first one is used to encrypt new tickets and
	// all are tried to decrypt tickets.
	ticketKeys []ticketKey

	// ClientFinishedIsFirst is true if the client sent the first Finished
	// message during the most recent handshake. This is recorded because
	// the first transmitted Finished message is the tls-unique
	// channel-binding value.
	ClientFinishedIsFirst bool `json:"client_finished_is_first"`

	// closeNotifyErr is any error from sending the alertCloseNotify record.
	closeNotifyErr error
	// CloseNotifySent is true if the conn attempted to send an
	// alertCloseNotify record.
	CloseNotifySent bool `json:"close_notify_sent"`

	// ClientFinished and ServerFinished contain the Finished message sent
	// by the client or server In the most recent handshake. This is
	// retained to support the renegotiation extension and tls-unique
	// channel-binding.
	ClientFinished [12]byte `json:"client_finished"`
	ServerFinished [12]byte `json:"server_finished"`

	// ClientProtocol is the negotiated ALPN protocol.
	ClientProtocol string `json:"client_protocol"`

	// input/output
	In halfConn `json:"in_halfconn"`
	Out halfConn `json:"out_halfconn"`
	// Todo - need to save these and restore manually
	rawInput  bytes.Buffer // Raw input, starting with a record header
	// Todo - need to save these and restore manually
	input     bytes.Reader // application Data waiting to be read, from rawInput.Next
	// Todo - need to save these and restore manually
	hand      bytes.Buffer // handshake Data waiting to be read
	Buffering bool `json:"buffering"`        // whether records are buffered In SendBuf
	SendBuf   []byte `json:"send_buf"`      // a buffer of records waiting to be sent

	// BytesSent counts the bytes of application Data sent.
	// PacketsSent counts packets.
	BytesSent   int64 `json:"bytes_sent"`
	PacketsSent int64 `json:"packets_sent"`

	// RetryCount counts the number of consecutive non-advancing records
	// received by conn.readRecord. That is, records that neither advance the
	// handshake, nor deliver application Data. Protected by In.Mutex.
	RetryCount int `json:"retry_count"`

	// ActiveCall is an atomic int32; the low bit is whether Close has
	// been called. the rest of the bits are the number of goroutines
	// In Conn.Write.
	ActiveCall int32 `json:"active_call"`

	Tmp [16]byte `json:"tmp"`

	HS serializedHS `json:"hs"`
	InHalfConn serializedHalfConn `json:"in_half_conn"`
	OutHalfConn serializedHalfConn `json:"out_half_conn"`

	ReadBuf []byte `json:"read_buf"`
	ReadBuffOffset int `json:"read_buff_offset"`
	ClientHello clientHelloMsg `json:"client_hello"`

}

type serializedHS struct {
	MasterSecret []byte `json:"master_secret"`
	H [8]uint32 `json:"h"`
	X [64]uint8 `json:"x"`
	Nx int `json:"nx"`
	Len uint64 `json:"len"`
	Is224 bool `json:"is_224"`
	ClientFinished []byte `json:"client_finished"`
	Suite cipherSuiteTLS13 `json:"suite"`
	TrafficSecret []byte `json:"traffic_secret"`
}

type serializedHalfConn struct {
	NonceMask [12]uint8
	AEADAesGCM serializedAEADAESGCM
	AEADChaCha serializedAEADChaCha
}

type serializedAEADAESGCM struct {
	Ks []uint32 `json:"ks"`
	ProductTable [256]uint8 `json:"product_table"`
	NonceSize int `json:"nonce_size"`
	TagSize int `json:"tag_size"`
}

type serializedAEADChaCha struct {
	Key [32]uint8 `json:"key"`
}

// Access to net.conn methods.
// Cannot just embed net.conn because that would
// export the struct field too.

// LocalAddr returns the local network address.
func (c *Conn) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

// RemoteAddr returns the remote network address.
func (c *Conn) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

// SetDeadline sets the read and write deadlines associated with the connection.
// A zero value for t means Read and Write will not time Out.
// After a Write has timed Out, the TLS state is corrupt and all future writes will return the same error.
func (c *Conn) SetDeadline(t time.Time) error {
	return c.conn.SetDeadline(t)
}

// SetReadDeadline sets the read deadline on the underlying connection.
// A zero value for t means Read will not time Out.
func (c *Conn) SetReadDeadline(t time.Time) error {
	return c.conn.SetReadDeadline(t)
}

// SetWriteDeadline sets the write deadline on the underlying connection.
// A zero value for t means Write will not time Out.
// After a Write has timed Out, the TLS state is corrupt and all future writes will return the same error.
func (c *Conn) SetWriteDeadline(t time.Time) error {
	return c.conn.SetWriteDeadline(t)
}

// A halfConn represents one direction of the record layer
// connection, either sending or receiving.
type halfConn struct {
	sync.Mutex

	Err     error       `json:"err"`      // first permanent error
	Version uint16      `json:"version"`     // protocol Version
	Seq     [8]byte `json:"seq"` // 64-bit sequence number

	Cipher  interface{} `json:"cipher"` // Cipher algorithm
	mac     hash.Hash

	scratchBuf [13]byte // to avoid allocs; interface method args escape

	nextCipher interface{} // next encryption state
	nextMac    hash.Hash   // next MAC algorithm

	TrafficSecret []byte `json:"traffic_secret"` // current TLS 1.3 traffic secret
	Iv []byte `json:"iv"`
}

type permanentError struct {
	err net.Error
}

func (e *permanentError) Error() string   { return e.err.Error() }
func (e *permanentError) Unwrap() error   { return e.err }
func (e *permanentError) Timeout() bool   { return e.err.Timeout() }
func (e *permanentError) Temporary() bool { return false }

func (hc *halfConn) setErrorLocked(err error) error {
	if e, ok := err.(net.Error); ok {
		hc.Err = &permanentError{err: e}
	} else {
		hc.Err = err
	}
	return hc.Err
}

// prepareCipherSpec sets the encryption and MAC states
// that a subsequent changeCipherSpec will use.
func (hc *halfConn) prepareCipherSpec(version uint16, cipher interface{}, mac hash.Hash) {
	hc.Version = version
	hc.nextCipher = cipher
	hc.nextMac = mac
}

// changeCipherSpec changes the encryption and MAC states
// to the ones previously passed to prepareCipherSpec.
func (hc *halfConn) changeCipherSpec() error {
	if hc.nextCipher == nil || hc.Version == VersionTLS13 {
		return alertInternalError
	}
	hc.Cipher = hc.nextCipher
	hc.mac = hc.nextMac
	hc.nextCipher = nil
	hc.nextMac = nil
	for i := range hc.Seq {
		hc.Seq[i] = 0
	}
	return nil
}

func (hc *halfConn) setTrafficSecret(suite *cipherSuiteTLS13, secret []byte) {
	hc.TrafficSecret = secret
	key, iv := suite.trafficKey(secret)
	hc.Iv = iv
	hc.Cipher = suite.aead(key, iv)
	for i := range hc.Seq {
		hc.Seq[i] = 0
	}
}

// incSeq increments the sequence number.
func (hc *halfConn) incSeq() {
	for i := 7; i >= 0; i-- {
		hc.Seq[i]++
		if hc.Seq[i] != 0 {
			return
		}
	}

	// Not allowed to let sequence number wrap.
	// Instead, must renegotiate before it does.
	// Not likely enough to bother.
	panic("TLS: sequence number wraparound")
}

// explicitNonceLen returns the number of bytes of explicit nonce or IV included
// In each record. Explicit nonces are present only In CBC modes after TLS 1.0
// and In certain AEAD modes In TLS 1.2.
func (hc *halfConn) explicitNonceLen() int {
	if hc.Cipher == nil {
		return 0
	}

	switch c := hc.Cipher.(type) {
	case cipher.Stream:
		return 0
	case aead:
		return c.explicitNonceLen()
	case cbcMode:
		// TLS 1.1 introduced a per-record explicit IV to fix the BEAST attack.
		if hc.Version >= VersionTLS11 {
			return c.BlockSize()
		}
		return 0
	default:
		panic("unknown Cipher type")
	}
}

// extractPadding returns, In constant time, the length of the padding to remove
// from the end of payload. It also returns a byte which is equal to 255 if the
// padding was valid and 0 otherwise. See RFC 2246, Section 6.2.3.2.
func extractPadding(payload []byte) (toRemove int, good byte) {
	if len(payload) < 1 {
		return 0, 0
	}

	paddingLen := payload[len(payload)-1]
	t := uint(len(payload)-1) - uint(paddingLen)
	// if len(payload) >= (paddingLen - 1) then the MSB of t is zero
	good = byte(int32(^t) >> 31)

	// The maximum possible padding length plus the actual length field
	toCheck := 256
	// The length of the padded Data is public, so we can use an if here
	if toCheck > len(payload) {
		toCheck = len(payload)
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

	// Zero the padding length on error. This ensures any unchecked bytes
	// are included In the MAC. Otherwise, an attacker that could
	// distinguish MAC failures from padding failures could mount an attack
	// similar to POODLE In SSL 3.0: given a good ciphertext that uses a
	// full block's worth of padding, replace the final block with another
	// block. If the MAC check passed but the padding check failed, the
	// last byte of that block decrypted to the block size.
	//
	// See also macAndPaddingGood logic below.
	paddingLen &= good

	toRemove = int(paddingLen) + 1
	return
}

func roundUp(a, b int) int {
	return a + (b-a%b)%b
}

// cbcMode is an interface for block ciphers using Cipher block chaining.
type cbcMode interface {
	cipher.BlockMode
	SetIV([]byte)
}

// decrypt authenticates and decrypts the record if protection is active at
// this stage. The returned plaintext might overlap with the input.
func (hc *halfConn) decrypt(record []byte) ([]byte, recordType, error) {
	var plaintext []byte
	typ := recordType(record[0])
	payload := record[recordHeaderLen:]

	// In TLS 1.3, change_cipher_spec messages are to be ignored without being
	// decrypted. See RFC 8446, Appendix D.4.
	if hc.Version == VersionTLS13 && typ == recordTypeChangeCipherSpec {
		return payload, typ, nil
	}

	paddingGood := byte(255)
	paddingLen := 0

	explicitNonceLen := hc.explicitNonceLen()

	if hc.Cipher != nil {
		switch c := hc.Cipher.(type) {
		case cipher.Stream:
			c.XORKeyStream(payload, payload)
		case aead:
			if len(payload) < explicitNonceLen {
				return nil, 0, alertBadRecordMAC
			}
			nonce := payload[:explicitNonceLen]
			if len(nonce) == 0 {
				nonce = hc.Seq[:]
			}
			payload = payload[explicitNonceLen:]

			var additionalData []byte
			if hc.Version == VersionTLS13 {
				additionalData = record[:recordHeaderLen]
			} else {
				additionalData = append(hc.scratchBuf[:0], hc.Seq[:]...)
				additionalData = append(additionalData, record[:3]...)
				n := len(payload) - c.Overhead()
				additionalData = append(additionalData, byte(n>>8), byte(n))
			}

			var err error
			plaintext, err = c.Open(payload[:0], nonce, payload, additionalData)
			if err != nil {
				return nil, 0, alertBadRecordMAC
			}
		case cbcMode:
			blockSize := c.BlockSize()
			minPayload := explicitNonceLen + roundUp(hc.mac.Size()+1, blockSize)
			if len(payload)%blockSize != 0 || len(payload) < minPayload {
				return nil, 0, alertBadRecordMAC
			}

			if explicitNonceLen > 0 {
				c.SetIV(payload[:explicitNonceLen])
				payload = payload[explicitNonceLen:]
			}
			c.CryptBlocks(payload, payload)

			// In a limited attempt to protect against CBC padding oracles like
			// Lucky13, the Data past paddingLen (which is secret) is passed to
			// the MAC function as extra Data, to be fed into the HMAC after
			// computing the digest. This makes the MAC roughly constant time as
			// long as the digest computation is constant time and does not
			// affect the subsequent write, modulo cache effects.
			paddingLen, paddingGood = extractPadding(payload)
		default:
			panic("unknown Cipher type")
		}

		if hc.Version == VersionTLS13 {
			if typ != recordTypeApplicationData {
				return nil, 0, alertUnexpectedMessage
			}
			if len(plaintext) > maxPlaintext+1 {
				return nil, 0, alertRecordOverflow
			}
			// Remove padding and find the ContentType scanning from the end.
			for i := len(plaintext) - 1; i >= 0; i-- {
				if plaintext[i] != 0 {
					typ = recordType(plaintext[i])
					plaintext = plaintext[:i]
					break
				}
				if i == 0 {
					return nil, 0, alertUnexpectedMessage
				}
			}
		}
	} else {
		plaintext = payload
	}

	if hc.mac != nil {
		macSize := hc.mac.Size()
		if len(payload) < macSize {
			return nil, 0, alertBadRecordMAC
		}

		n := len(payload) - macSize - paddingLen
		n = subtle.ConstantTimeSelect(int(uint32(n)>>31), 0, n) // if n < 0 { n = 0 }
		record[3] = byte(n >> 8)
		record[4] = byte(n)
		remoteMAC := payload[n : n+macSize]
		localMAC := tls10MAC(hc.mac, hc.scratchBuf[:0], hc.Seq[:], record[:recordHeaderLen], payload[:n], payload[n+macSize:])

		// This is equivalent to checking the MACs and paddingGood
		// separately, but In constant-time to prevent distinguishing
		// padding failures from MAC failures. Depending on what value
		// of paddingLen was returned on bad padding, distinguishing
		// bad MAC from bad padding can lead to an attack.
		//
		// See also the logic at the end of extractPadding.
		macAndPaddingGood := subtle.ConstantTimeCompare(localMAC, remoteMAC) & int(paddingGood)
		if macAndPaddingGood != 1 {
			return nil, 0, alertBadRecordMAC
		}

		plaintext = payload[:n]
	}

	hc.incSeq()
	return plaintext, typ, nil
}

// sliceForAppend extends the input slice by n bytes. head is the full extended
// slice, while tail is the appended part. If the original slice has sufficient
// capacity no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

// encrypt encrypts payload, adding the appropriate nonce and/or MAC, and
// appends it to record, which must already contain the record header.
func (hc *halfConn) encrypt(record, payload []byte, rand io.Reader) ([]byte, error) {
	if hc.Cipher == nil {
		return append(record, payload...), nil
	}

	var explicitNonce []byte
	if explicitNonceLen := hc.explicitNonceLen(); explicitNonceLen > 0 {
		record, explicitNonce = sliceForAppend(record, explicitNonceLen)
		if _, isCBC := hc.Cipher.(cbcMode); !isCBC && explicitNonceLen < 16 {
			// The AES-GCM construction In TLS has an explicit nonce so that the
			// nonce can be Random. However, the nonce is only 8 bytes which is
			// too small for a secure, Random nonce. Therefore we use the
			// sequence number as the nonce. The 3DES-CBC construction also has
			// an 8 bytes nonce but its nonces must be unpredictable (see RFC
			// 5246, Appendix F.3), forcing us to use randomness. That's not
			// 3DES' biggest problem anyway because the birthday bound on block
			// collision is reached first due to its similarly small block size
			// (see the Sweet32 attack).
			copy(explicitNonce, hc.Seq[:])
		} else {
			if _, err := io.ReadFull(rand, explicitNonce); err != nil {
				return nil, err
			}
		}
	}

	var dst []byte
	switch c := hc.Cipher.(type) {
	case cipher.Stream:
		mac := tls10MAC(hc.mac, hc.scratchBuf[:0], hc.Seq[:], record[:recordHeaderLen], payload, nil)
		record, dst = sliceForAppend(record, len(payload)+len(mac))
		c.XORKeyStream(dst[:len(payload)], payload)
		c.XORKeyStream(dst[len(payload):], mac)
	case aead:
		nonce := explicitNonce
		if len(nonce) == 0 {
			nonce = hc.Seq[:]
		}

		if hc.Version == VersionTLS13 {
			record = append(record, payload...)

			// Encrypt the actual ContentType and replace the plaintext one.
			record = append(record, record[0])
			record[0] = byte(recordTypeApplicationData)

			n := len(payload) + 1 + c.Overhead()
			record[3] = byte(n >> 8)
			record[4] = byte(n)

			record = c.Seal(record[:recordHeaderLen],
				nonce, record[recordHeaderLen:], record[:recordHeaderLen])
		} else {
			additionalData := append(hc.scratchBuf[:0], hc.Seq[:]...)
			additionalData = append(additionalData, record[:recordHeaderLen]...)
			record = c.Seal(record, nonce, payload, additionalData)
		}
	case cbcMode:
		mac := tls10MAC(hc.mac, hc.scratchBuf[:0], hc.Seq[:], record[:recordHeaderLen], payload, nil)
		blockSize := c.BlockSize()
		plaintextLen := len(payload) + len(mac)
		paddingLen := blockSize - plaintextLen%blockSize
		record, dst = sliceForAppend(record, plaintextLen+paddingLen)
		copy(dst, payload)
		copy(dst[len(payload):], mac)
		for i := plaintextLen; i < len(dst); i++ {
			dst[i] = byte(paddingLen - 1)
		}
		if len(explicitNonce) > 0 {
			c.SetIV(explicitNonce)
		}
		c.CryptBlocks(dst, dst)
	default:
		panic("unknown Cipher type")
	}

	// Update length to include nonce, MAC and any block padding needed.
	n := len(record) - recordHeaderLen
	record[3] = byte(n >> 8)
	record[4] = byte(n)
	hc.incSeq()

	return record, nil
}

// RecordHeaderError is returned when a TLS record header is invalid.
type RecordHeaderError struct {
	// Msg contains a human readable string that describes the error.
	Msg string
	// RecordHeader contains the five bytes of TLS record header that
	// triggered the error.
	RecordHeader [5]byte
	// Conn provides the underlying net.Conn In the case that a client
	// sent an initial handshake that didn't look like TLS.
	// It is nil if there's already been a handshake or a TLS alert has
	// been written to the connection.
	Conn net.Conn
}

func (e RecordHeaderError) Error() string { return "tls: " + e.Msg }

func (c *Conn) newRecordHeaderError(conn net.Conn, msg string) (err RecordHeaderError) {
	err.Msg = msg
	err.Conn = conn
	copy(err.RecordHeader[:], c.rawInput.Bytes())
	return err
}

func (c *Conn) readRecord() error {
	return c.readRecordOrCCS(false)
}

func (c *Conn) readChangeCipherSpec() error {
	return c.readRecordOrCCS(true)
}

// readRecordOrCCS reads one or more TLS records from the connection and
// updates the record layer state. Some invariants:
//   * c.In must be locked
//   * c.input must be empty
// During the handshake one and only one of the following will happen:
//   - c.hand grows
//   - c.In.changeCipherSpec is called
//   - an error is returned
// After the handshake one and only one of the following will happen:
//   - c.hand grows
//   - c.input is set
//   - an error is returned
func (c *Conn) readRecordOrCCS(expectChangeCipherSpec bool) error {
	if c.In.Err != nil {
		return c.In.Err
	}
	handshakeComplete := c.handshakeComplete()

	// This function modifies c.rawInput, which owns the c.input memory.
	if c.input.Len() != 0 {
		return c.In.setErrorLocked(errors.New("tls: internal error: attempted to read record with pending application Data"))
	}
	c.input.Reset(nil)

	// Read header, payload.
	if err := c.readFromUntil(c.conn, recordHeaderLen); err != nil {
		// RFC 8446, Section 6.1 suggests that EOF without an alertCloseNotify
		// is an error, but popular web sites seem to do this, so we accept it
		// if and only if at the record boundary.
		if err == io.ErrUnexpectedEOF && c.rawInput.Len() == 0 {
			err = io.EOF
		}
		if e, ok := err.(net.Error); !ok || !e.Temporary() {
			c.In.setErrorLocked(err)
		}
		return err
	}
	hdr := c.rawInput.Bytes()[:recordHeaderLen]
	typ := recordType(hdr[0])

	// No valid TLS record has a type of 0x80, however SSLv2 Handshakes
	// start with a uint16 length where the MSB is set and the first record
	// is always < 256 bytes long. Therefore typ == 0x80 strongly suggests
	// an SSLv2 client.
	if !handshakeComplete && typ == 0x80 {
		c.sendAlert(alertProtocolVersion)
		return c.In.setErrorLocked(c.newRecordHeaderError(nil, "unsupported SSLv2 handshake received"))
	}

	vers := uint16(hdr[1])<<8 | uint16(hdr[2])
	n := int(hdr[3])<<8 | int(hdr[4])
	if c.HaveVers && c.Vers != VersionTLS13 && vers != c.Vers {
		c.sendAlert(alertProtocolVersion)
		msg := fmt.Sprintf("received record with Version %x when expecting Version %x", vers, c.Vers)
		return c.In.setErrorLocked(c.newRecordHeaderError(nil, msg))
	}
	if !c.HaveVers {
		// First message, be extra suspicious: this might not be a TLS
		// client. Bail Out before reading a full 'body', if possible.
		// The current max Version is 3.3 so if the Version is >= 16.0,
		// it's probably not real.
		if (typ != recordTypeAlert && typ != recordTypeHandshake) || vers >= 0x1000 {
			return c.In.setErrorLocked(c.newRecordHeaderError(c.conn, "first record does not look like a TLS handshake"))
		}
	}
	if c.Vers == VersionTLS13 && n > maxCiphertextTLS13 || n > maxCiphertext {
		c.sendAlert(alertRecordOverflow)
		msg := fmt.Sprintf("oversized record received with length %d", n)
		return c.In.setErrorLocked(c.newRecordHeaderError(nil, msg))
	}
	if err := c.readFromUntil(c.conn, recordHeaderLen+n); err != nil {
		if e, ok := err.(net.Error); !ok || !e.Temporary() {
			c.In.setErrorLocked(err)
		}
		return err
	}

	// Process message.
	record := c.rawInput.Next(recordHeaderLen + n)
	data, typ, err := c.In.decrypt(record)
	if err != nil {
		return c.In.setErrorLocked(c.sendAlert(err.(alert)))
	}
	if len(data) > maxPlaintext {
		return c.In.setErrorLocked(c.sendAlert(alertRecordOverflow))
	}

	// Application Data messages are always protected.
	if c.In.Cipher == nil && typ == recordTypeApplicationData {
		return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
	}

	if typ != recordTypeAlert && typ != recordTypeChangeCipherSpec && len(data) > 0 {
		// This is a state-advancing message: reset the retry count.
		c.RetryCount = 0
	}

	// Handshake messages MUST NOT be interleaved with other record types In TLS 1.3.
	if c.Vers == VersionTLS13 && typ != recordTypeHandshake && c.hand.Len() > 0 {
		return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
	}

	switch typ {
	default:
		return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))

	case recordTypeAlert:
		if len(data) != 2 {
			return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
		}
		if alert(data[1]) == alertCloseNotify {
			return c.In.setErrorLocked(io.EOF)
		}
		if c.Vers == VersionTLS13 {
			return c.In.setErrorLocked(&net.OpError{Op: "remote error", Err: alert(data[1])})
		}
		switch data[0] {
		case alertLevelWarning:
			// Drop the record on the floor and retry.
			return c.retryReadRecord(expectChangeCipherSpec)
		case alertLevelError:
			return c.In.setErrorLocked(&net.OpError{Op: "remote error", Err: alert(data[1])})
		default:
			return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
		}

	case recordTypeChangeCipherSpec:
		if len(data) != 1 || data[0] != 1 {
			return c.In.setErrorLocked(c.sendAlert(alertDecodeError))
		}
		// Handshake messages are not allowed to fragment across the CCS.
		if c.hand.Len() > 0 {
			return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
		}
		// In TLS 1.3, change_cipher_spec records are ignored until the
		// Finished. See RFC 8446, Appendix D.4. Note that according to Section
		// 5, a server can send a ChangeCipherSpec before its ServerHello, when
		// c.Vers is still unset. That's not useful though and suspicious if the
		// server then selects a lower protocol Version, so don't allow that.
		if c.Vers == VersionTLS13 {
			return c.retryReadRecord(expectChangeCipherSpec)
		}
		if !expectChangeCipherSpec {
			return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
		}
		if err := c.In.changeCipherSpec(); err != nil {
			return c.In.setErrorLocked(c.sendAlert(err.(alert)))
		}

	case recordTypeApplicationData:
		if !handshakeComplete || expectChangeCipherSpec {
			return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
		}
		// Some OpenSSL servers send empty records In order to randomize the
		// CBC IV. Ignore a limited number of empty records.
		if len(data) == 0 {
			return c.retryReadRecord(expectChangeCipherSpec)
		}
		// Note that Data is owned by c.rawInput, following the Next call above,
		// to avoid copying the plaintext. This is safe because c.rawInput is
		// not read from or written to until c.input is drained.
		c.input.Reset(data)

	case recordTypeHandshake:
		if len(data) == 0 || expectChangeCipherSpec {
			return c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
		}
		c.hand.Write(data)
	}

	return nil
}

// retryReadRecord recurses into readRecordOrCCS to drop a non-advancing record, like
// a warning alert, empty application_data, or a change_cipher_spec In TLS 1.3.
func (c *Conn) retryReadRecord(expectChangeCipherSpec bool) error {
	c.RetryCount++
	if c.RetryCount > maxUselessRecords {
		c.sendAlert(alertUnexpectedMessage)
		return c.In.setErrorLocked(errors.New("tls: too many ignored records"))
	}
	return c.readRecordOrCCS(expectChangeCipherSpec)
}

// atLeastReader reads from R, stopping with EOF once at least N bytes have been
// read. It is different from an io.LimitedReader In that it doesn't cut short
// the last Read call, and In that it considers an early EOF an error.
type atLeastReader struct {
	R io.Reader
	N int64
}

func (r *atLeastReader) Read(p []byte) (int, error) {
	if r.N <= 0 {
		return 0, io.EOF
	}
	n, err := r.R.Read(p)
	r.N -= int64(n) // won't underflow unless len(p) >= n > 9223372036854775809
	if r.N > 0 && err == io.EOF {
		return n, io.ErrUnexpectedEOF
	}
	if r.N <= 0 && err == nil {
		return n, io.EOF
	}
	return n, err
}

// readFromUntil reads from r into c.rawInput until c.rawInput contains
// at least n bytes or else returns an error.
func (c *Conn) readFromUntil(r io.Reader, n int) error {
	if c.rawInput.Len() >= n {
		return nil
	}
	needs := n - c.rawInput.Len()
	// There might be extra input waiting on the wire. Make a best effort
	// attempt to fetch it so that it can be used In (*conn).Read to
	// "predict" closeNotify alerts.
	c.rawInput.Grow(needs + bytes.MinRead)
	_, err := c.rawInput.ReadFrom(&atLeastReader{r, int64(needs)})
	return err
}

// sendAlert sends a TLS alert message.
func (c *Conn) sendAlertLocked(err alert) error {
	switch err {
	case alertNoRenegotiation, alertCloseNotify:
		c.Tmp[0] = alertLevelWarning
	default:
		c.Tmp[0] = alertLevelError
	}
	c.Tmp[1] = byte(err)

	_, writeErr := c.writeRecordLocked(recordTypeAlert, c.Tmp[0:2])
	if err == alertCloseNotify {
		// closeNotify is a special case In that it isn't an error.
		return writeErr
	}

	return c.Out.setErrorLocked(&net.OpError{Op: "local error", Err: err})
}

// sendAlert sends a TLS alert message.
func (c *Conn) sendAlert(err alert) error {
	c.Out.Lock()
	defer c.Out.Unlock()
	return c.sendAlertLocked(err)
}

const (
	// tcpMSSEstimate is a conservative estimate of the TCP maximum segment
	// size (MSS). A constant is used, rather than querying the kernel for
	// the actual MSS, to avoid complexity. The value here is the IPv6
	// minimum MTU (1280 bytes) minus the overhead of an IPv6 header (40
	// bytes) and a TCP header with timestamps (32 bytes).
	tcpMSSEstimate = 1208

	// recordSizeBoostThreshold is the number of bytes of application Data
	// sent after which the TLS record size will be increased to the
	// maximum.
	recordSizeBoostThreshold = 128 * 1024
)

// maxPayloadSizeForWrite returns the maximum TLS payload size to use for the
// next application Data record. There is the following trade-off:
//
//   - For latency-sensitive applications, such as web browsing, each TLS
//     record should fit In one TCP segment.
//   - For throughput-sensitive applications, such as large file transfers,
//     larger TLS records better amortize framing and encryption overheads.
//
// A simple heuristic that works well In practice is to use small records for
// the first 1MB of Data, then use larger records for subsequent Data, and
// reset back to smaller records after the connection becomes idle. See "High
// Performance Web Networking", Chapter 4, or:
// https://www.igvita.com/2013/10/24/optimizing-tls-record-size-and-buffering-latency/
//
// In the interests of simplicity and determinism, this code does not attempt
// to reset the record size once the connection is idle, however.
func (c *Conn) maxPayloadSizeForWrite(typ recordType) int {
	if c.config.DynamicRecordSizingDisabled || typ != recordTypeApplicationData {
		return maxPlaintext
	}

	if c.BytesSent >= recordSizeBoostThreshold {
		return maxPlaintext
	}

	// Subtract TLS overheads to get the maximum payload size.
	payloadBytes := tcpMSSEstimate - recordHeaderLen - c.Out.explicitNonceLen()
	if c.Out.Cipher != nil {
		switch ciph := c.Out.Cipher.(type) {
		case cipher.Stream:
			payloadBytes -= c.Out.mac.Size()
		case cipher.AEAD:
			payloadBytes -= ciph.Overhead()
		case cbcMode:
			blockSize := ciph.BlockSize()
			// The payload must fit In a multiple of blockSize, with
			// room for at least one padding byte.
			payloadBytes = (payloadBytes & ^(blockSize - 1)) - 1
			// The MAC is appended before padding so affects the
			// payload size directly.
			payloadBytes -= c.Out.mac.Size()
		default:
			panic("unknown Cipher type")
		}
	}
	if c.Vers == VersionTLS13 {
		payloadBytes-- // encrypted ContentType
	}

	// Allow packet growth In arithmetic progression up to max.
	pkt := c.PacketsSent
	c.PacketsSent++
	if pkt > 1000 {
		return maxPlaintext // avoid overflow In multiply below
	}

	n := payloadBytes * int(pkt+1)
	if n > maxPlaintext {
		n = maxPlaintext
	}
	return n
}

func (c *Conn) write(data []byte) (int, error) {
	if c.Buffering {
		c.SendBuf = append(c.SendBuf, data...)
		return len(data), nil
	}

	n, err := c.conn.Write(data)
	c.BytesSent += int64(n)
	return n, err
}

func (c *Conn) flush() (int, error) {
	if len(c.SendBuf) == 0 {
		return 0, nil
	}

	n, err := c.conn.Write(c.SendBuf)
	c.BytesSent += int64(n)
	c.SendBuf = nil
	c.Buffering = false
	return n, err
}

// outBufPool pools the record-sized scratch buffers used by writeRecordLocked.
var outBufPool = sync.Pool{
	New: func() interface{} {
		return new([]byte)
	},
}

// writeRecordLocked writes a TLS record with the given type and payload to the
// connection and updates the record layer state.
func (c *Conn) writeRecordLocked(typ recordType, data []byte) (int, error) {
	outBufPtr := outBufPool.Get().(*[]byte)
	outBuf := *outBufPtr
	defer func() {
		// You might be tempted to simplify this by just passing &outBuf to Put,
		// but that would make the local copy of the outBuf slice header escape
		// to the heap, causing an allocation. Instead, we keep around the
		// pointer to the slice header returned by Get, which is already on the
		// heap, and overwrite and return that.
		*outBufPtr = outBuf
		outBufPool.Put(outBufPtr)
	}()

	var n int
	for len(data) > 0 {
		m := len(data)
		if maxPayload := c.maxPayloadSizeForWrite(typ); m > maxPayload {
			m = maxPayload
		}

		_, outBuf = sliceForAppend(outBuf[:0], recordHeaderLen)
		outBuf[0] = byte(typ)
		vers := c.Vers
		if vers == 0 {
			// Some TLS servers fail if the record Version is
			// greater than TLS 1.0 for the initial ClientHello.
			vers = VersionTLS10
		} else if vers == VersionTLS13 {
			// TLS 1.3 froze the record layer Version to 1.2.
			// See RFC 8446, Section 5.1.
			vers = VersionTLS12
		}
		outBuf[1] = byte(vers >> 8)
		outBuf[2] = byte(vers)
		outBuf[3] = byte(m >> 8)
		outBuf[4] = byte(m)

		var err error
		outBuf, err = c.Out.encrypt(outBuf, data[:m], c.config.rand())
		if err != nil {
			return n, err
		}
		if _, err := c.write(outBuf); err != nil {
			return n, err
		}
		n += m
		data = data[m:]
	}

	if typ == recordTypeChangeCipherSpec && c.Vers != VersionTLS13 {
		if err := c.Out.changeCipherSpec(); err != nil {
			return n, c.sendAlertLocked(err.(alert))
		}
	}

	return n, nil
}

// writeRecord writes a TLS record with the given type and payload to the
// connection and updates the record layer state.
func (c *Conn) writeRecord(typ recordType, data []byte) (int, error) {
	c.Out.Lock()
	defer c.Out.Unlock()

	return c.writeRecordLocked(typ, data)
}

// readHandshake reads the next handshake message from
// the record layer.
func (c *Conn) readHandshake() (interface{}, error) {
	for c.hand.Len() < 4 {
		if err := c.readRecord(); err != nil {
			return nil, err
		}
	}

	data := c.hand.Bytes()
	n := int(data[1])<<16 | int(data[2])<<8 | int(data[3])
	if n > maxHandshake {
		c.sendAlertLocked(alertInternalError)
		return nil, c.In.setErrorLocked(fmt.Errorf("tls: handshake message of length %d bytes exceeds maximum of %d bytes", n, maxHandshake))
	}
	for c.hand.Len() < 4+n {
		if err := c.readRecord(); err != nil {
			return nil, err
		}
	}
	data = c.hand.Next(4 + n)
	var m handshakeMessage
	switch data[0] {
	case typeHelloRequest:
		m = new(helloRequestMsg)
	case typeClientHello:
		m = new(clientHelloMsg)
	case typeServerHello:
		m = new(serverHelloMsg)
	case typeNewSessionTicket:
		if c.Vers == VersionTLS13 {
			m = new(newSessionTicketMsgTLS13)
		} else {
			m = new(newSessionTicketMsg)
		}
	case typeCertificate:
		if c.Vers == VersionTLS13 {
			m = new(certificateMsgTLS13)
		} else {
			m = new(certificateMsg)
		}
	case typeCertificateRequest:
		if c.Vers == VersionTLS13 {
			m = new(certificateRequestMsgTLS13)
		} else {
			m = &certificateRequestMsg{
				hasSignatureAlgorithm: c.Vers >= VersionTLS12,
			}
		}
	case typeCertificateStatus:
		m = new(certificateStatusMsg)
	case typeServerKeyExchange:
		m = new(serverKeyExchangeMsg)
	case typeServerHelloDone:
		m = new(serverHelloDoneMsg)
	case typeClientKeyExchange:
		m = new(clientKeyExchangeMsg)
	case typeCertificateVerify:
		m = &certificateVerifyMsg{
			hasSignatureAlgorithm: c.Vers >= VersionTLS12,
		}
	case typeFinished:
		m = new(finishedMsg)
	case typeEncryptedExtensions:
		m = new(encryptedExtensionsMsg)
	case typeEndOfEarlyData:
		m = new(endOfEarlyDataMsg)
	case typeKeyUpdate:
		m = new(keyUpdateMsg)
	default:
		return nil, c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
	}

	// The handshake message unmarshalers
	// expect to be able to keep references to Data,
	// so pass In a fresh copy that won't be overwritten.
	data = append([]byte(nil), data...)

	if !m.unmarshal(data) {
		return nil, c.In.setErrorLocked(c.sendAlert(alertUnexpectedMessage))
	}
	return m, nil
}

var (
	errShutdown = errors.New("tls: protocol is shutdown")
)

// Write writes Data to the connection.
//
// As Write calls Handshake, In order to prevent indefinite blocking a deadline
// must be set for both Read and Write before Write is called when the handshake
// has not yet completed. See SetDeadline, SetReadDeadline, and
// SetWriteDeadline.
func (c *Conn) Write(b []byte) (int, error) {
	// interlock with Close below
	for {
		x := atomic.LoadInt32(&c.ActiveCall)
		if x&1 != 0 {
			return 0, net.ErrClosed
		}
		if atomic.CompareAndSwapInt32(&c.ActiveCall, x, x+2) {
			break
		}
	}
	defer atomic.AddInt32(&c.ActiveCall, -2)

	if err := c.Handshake(); err != nil {
		return 0, err
	}

	c.Out.Lock()
	defer c.Out.Unlock()

	if err := c.Out.Err; err != nil {
		return 0, err
	}

	if !c.handshakeComplete() {
		return 0, alertInternalError
	}

	if c.CloseNotifySent {
		return 0, errShutdown
	}

	// TLS 1.0 is susceptible to a chosen-plaintext
	// attack when using block mode ciphers due to predictable IVs.
	// This can be prevented by splitting each Application Data
	// record into two records, effectively randomizing the IV.
	//
	// https://www.openssl.org/~bodo/tls-cbc.txt
	// https://bugzilla.mozilla.org/show_bug.cgi?id=665814
	// https://www.imperialviolet.org/2012/01/15/beastfollowup.html

	var m int
	if len(b) > 1 && c.Vers == VersionTLS10 {
		if _, ok := c.Out.Cipher.(cipher.BlockMode); ok {
			n, err := c.writeRecordLocked(recordTypeApplicationData, b[:1])
			if err != nil {
				return n, c.Out.setErrorLocked(err)
			}
			m, b = 1, b[1:]
		}
	}

	n, err := c.writeRecordLocked(recordTypeApplicationData, b)
	return n + m, c.Out.setErrorLocked(err)
}

// handleRenegotiation processes a HelloRequest handshake message.
func (c *Conn) handleRenegotiation() error {
	if c.Vers == VersionTLS13 {
		return errors.New("tls: internal error: unexpected renegotiation")
	}

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	helloReq, ok := msg.(*helloRequestMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(helloReq, msg)
	}

	if !c.IsClient {
		return c.sendAlert(alertNoRenegotiation)
	}

	switch c.config.Renegotiation {
	case RenegotiateNever:
		return c.sendAlert(alertNoRenegotiation)
	case RenegotiateOnceAsClient:
		if c.Handshakes > 1 {
			return c.sendAlert(alertNoRenegotiation)
		}
	case RenegotiateFreelyAsClient:
		// Ok.
	default:
		c.sendAlert(alertInternalError)
		return errors.New("tls: unknown Renegotiation value")
	}

	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()

	atomic.StoreUint32(&c.HandshakeStatus, 0)
	if c.handshakeErr = c.clientHandshake(context.Background()); c.handshakeErr == nil {
		c.Handshakes++
	}
	return c.handshakeErr
}

// handlePostHandshakeMessage processes a handshake message arrived after the
// handshake is complete. Up to TLS 1.2, it indicates the start of a renegotiation.
func (c *Conn) handlePostHandshakeMessage() error {
	if c.Vers != VersionTLS13 {
		return c.handleRenegotiation()
	}

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	c.RetryCount++
	if c.RetryCount > maxUselessRecords {
		c.sendAlert(alertUnexpectedMessage)
		return c.In.setErrorLocked(errors.New("tls: too many non-advancing records"))
	}

	switch msg := msg.(type) {
	case *newSessionTicketMsgTLS13:
		return c.handleNewSessionTicket(msg)
	case *keyUpdateMsg:
		return c.handleKeyUpdate(msg)
	default:
		c.sendAlert(alertUnexpectedMessage)
		return fmt.Errorf("tls: received unexpected handshake message of type %T", msg)
	}
}

func (c *Conn) handleKeyUpdate(keyUpdate *keyUpdateMsg) error {
	cipherSuite := cipherSuiteTLS13ByID(c.CipherSuite)
	if cipherSuite == nil {
		return c.In.setErrorLocked(c.sendAlert(alertInternalError))
	}

	newSecret := cipherSuite.nextTrafficSecret(c.In.TrafficSecret)
	c.In.setTrafficSecret(cipherSuite, newSecret)

	if keyUpdate.updateRequested {
		c.Out.Lock()
		defer c.Out.Unlock()

		msg := &keyUpdateMsg{}
		_, err := c.writeRecordLocked(recordTypeHandshake, msg.marshal())
		if err != nil {
			// Surface the error at the next write.
			c.Out.setErrorLocked(err)
			return nil
		}

		newSecret := cipherSuite.nextTrafficSecret(c.Out.TrafficSecret)
		c.Out.setTrafficSecret(cipherSuite, newSecret)
	}

	return nil
}

// Read reads Data from the connection.
//
// As Read calls Handshake, In order to prevent indefinite blocking a deadline
// must be set for both Read and Write before Read is called when the handshake
// has not yet completed. See SetDeadline, SetReadDeadline, and
// SetWriteDeadline.
func (c *Conn) Read(b []byte) (int, error) {
	if err := c.Handshake(); err != nil {
		return 0, err
	}
	if len(b) == 0 {
		// Put this after Handshake, In case people were calling
		// Read(nil) for the side effect of the Handshake.
		return 0, nil
	}

	c.In.Lock()
	defer c.In.Unlock()

	for c.input.Len() == 0 {
		if err := c.readRecord(); err != nil {
			return 0, err
		}
		for c.hand.Len() > 0 {
			if err := c.handlePostHandshakeMessage(); err != nil {
				return 0, err
			}
		}
	}

	n, _ := c.input.Read(b)

	// If a close-notify alert is waiting, read it so that we can return (n,
	// EOF) instead of (n, nil), to signal to the HTTP response reading
	// goroutine that the connection is now closed. This eliminates a race
	// where the HTTP response reading goroutine would otherwise not observe
	// the EOF until its next read, by which time a client goroutine might
	// have already tried to reuse the HTTP connection for a new request.
	// See https://golang.org/cl/76400046 and https://golang.org/issue/3514
	if n != 0 && c.input.Len() == 0 && c.rawInput.Len() > 0 &&
		recordType(c.rawInput.Bytes()[0]) == recordTypeAlert {
		if err := c.readRecord(); err != nil {
			return n, err // will be io.EOF on closeNotify
		}
	}

	return n, nil
}

// Close closes the connection.
func (c *Conn) Close() error {
	// Interlock with conn.Write above.
	var x int32
	for {
		x = atomic.LoadInt32(&c.ActiveCall)
		if x&1 != 0 {
			return net.ErrClosed
		}
		if atomic.CompareAndSwapInt32(&c.ActiveCall, x, x|1) {
			break
		}
	}
	if x != 0 {
		// io.Writer and io.Closer should not be used concurrently.
		// If Close is called while a Write is currently In-flight,
		// interpret that as a sign that this Close is really just
		// being used to break the Write and/or clean up resources and
		// avoid sending the alertCloseNotify, which may block
		// waiting on handshakeMutex or the c.Out mutex.
		return c.conn.Close()
	}

	var alertErr error
	if c.handshakeComplete() {
		if err := c.closeNotify(); err != nil {
			alertErr = fmt.Errorf("tls: failed to send closeNotify alert (but connection was closed anyway): %w", err)
		}
	}

	if err := c.conn.Close(); err != nil {
		return err
	}
	return alertErr
}

var errEarlyCloseWrite = errors.New("tls: CloseWrite called before handshake complete")

// CloseWrite shuts down the writing side of the connection. It should only be
// called once the handshake has completed and does not call CloseWrite on the
// underlying connection. Most callers should just use Close.
func (c *Conn) CloseWrite() error {
	if !c.handshakeComplete() {
		return errEarlyCloseWrite
	}

	return c.closeNotify()
}

func (c *Conn) closeNotify() error {
	c.Out.Lock()
	defer c.Out.Unlock()

	if !c.CloseNotifySent {
		// Set a Write Deadline to prevent possibly blocking forever.
		c.SetWriteDeadline(time.Now().Add(time.Second * 5))
		c.closeNotifyErr = c.sendAlertLocked(alertCloseNotify)
		c.CloseNotifySent = true
		// Any subsequent writes will fail.
		c.SetWriteDeadline(time.Now())
	}
	return c.closeNotifyErr
}

// Handshake runs the client or server handshake
// protocol if it has not yet been run.
//
// Most uses of this package need not call Handshake explicitly: the
// first Read or Write will call it automatically.
//
// For control over canceling or setting a timeout on a handshake, use
// HandshakeContext or the Dialer's DialContext method instead.
func (c *Conn) Handshake() error {
	return c.HandshakeContext(context.Background())
}

// HandshakeContext runs the client or server handshake
// protocol if it has not yet been run.
//
// The provided Context must be non-nil. If the context is canceled before
// the handshake is complete, the handshake is interrupted and an error is returned.
// Once the handshake has completed, cancellation of the context will not affect the
// connection.
//
// Most uses of this package need not call HandshakeContext explicitly: the
// first Read or Write will call it automatically.
func (c *Conn) HandshakeContext(ctx context.Context) error {
	// Delegate to unexported method for named return
	// without confusing documented signature.
	return c.handshakeContext(ctx)
}

func (c *Conn) handshakeContext(ctx context.Context) (ret error) {
	handshakeCtx, cancel := context.WithCancel(ctx)
	// Note: defer this before starting the "interrupter" goroutine
	// so that we can tell the difference between the input being canceled and
	// this cancellation. In the former case, we need to close the connection.
	defer cancel()

	// Start the "interrupter" goroutine, if this context might be canceled.
	// (The background context cannot).
	//
	// The interrupter goroutine waits for the input context to be done and
	// closes the connection if this happens before the function returns.
	if ctx.Done() != nil {
		done := make(chan struct{})
		interruptRes := make(chan error, 1)
		defer func() {
			close(done)
			if ctxErr := <-interruptRes; ctxErr != nil {
				// Return context error to user.
				ret = ctxErr
			}
		}()
		go func() {
			select {
			case <-handshakeCtx.Done():
				// Close the connection, discarding the error
				_ = c.conn.Close()
				interruptRes <- handshakeCtx.Err()
			case <-done:
				interruptRes <- nil
			}
		}()
	}

	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()

	if err := c.handshakeErr; err != nil {
		return err
	}
	if c.handshakeComplete() {
		return nil
	}

	c.In.Lock()
	defer c.In.Unlock()

	c.handshakeErr = c.handshakeFn(handshakeCtx)
	if c.handshakeErr == nil {
		c.Handshakes++
	} else {
		// If an error occurred during the handshake try to flush the
		// alert that might be left In the buffer.
		c.flush()
	}

	if c.handshakeErr == nil && !c.handshakePrimaryComplete() {
		c.handshakeErr = errors.New("tls: internal error: handshake should have had a result")
	}

	return c.handshakeErr
}

// ConnectionState returns basic TLS details about the connection.
func (c *Conn) ConnectionState() ConnectionState {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()
	return c.connectionStateLocked()
}

func (c *Conn) connectionStateLocked() ConnectionState {
	var state ConnectionState
	state.HandshakeComplete = c.handshakeComplete()
	state.Version = c.Vers
	state.NegotiatedProtocol = c.ClientProtocol
	state.DidResume = c.DidResume
	state.NegotiatedProtocolIsMutual = true
	state.ServerName = c.ServerName
	state.CipherSuite = c.CipherSuite
	state.PeerCertificates = c.peerCertificates
	state.VerifiedChains = c.verifiedChains
	state.SignedCertificateTimestamps = c.Scts
	state.OCSPResponse = c.OcspResponse
	if !c.DidResume && c.Vers != VersionTLS13 {
		if c.ClientFinishedIsFirst {
			state.TLSUnique = c.ClientFinished[:]
		} else {
			state.TLSUnique = c.ServerFinished[:]
		}
	}
	if c.config.Renegotiation != RenegotiateNever {
		state.ekm = noExportedKeyingMaterial
	} else {
		state.ekm = c.ekm
	}
	return state
}

// OCSPResponse returns the stapled OCSP response from the TLS server, if
// any. (Only valid for client connections.)
func (c *Conn) OCSPResponse() []byte {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()

	return c.OcspResponse
}

// VerifyHostname checks that the peer certificate chain is valid for
// connecting to host. If so, it returns nil; if not, it returns an error
// describing the problem.
func (c *Conn) VerifyHostname(host string) error {
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()
	if !c.IsClient {
		return errors.New("tls: VerifyHostname called on TLS server connection")
	}
	if !c.handshakeComplete() {
		return errors.New("tls: handshake has not yet been performed")
	}
	if len(c.verifiedChains) == 0 {
		return errors.New("tls: handshake did not verify certificate chain")
	}
	return c.peerCertificates[0].VerifyHostname(host)
}

func (c *Conn) handshakeComplete() bool {
	return atomic.LoadUint32(&c.HandshakeStatus) == 1
}

func (c *Conn) handshakePrimaryComplete() bool {
	return atomic.LoadUint32(&c.HandshakePrimaryStatus) == 1
}
