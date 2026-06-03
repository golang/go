// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/sha256"
	"crypto/tls"
	"errors"
	"hash"

	"golang.org/x/crypto/chacha20"
	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/crypto/cryptobyte"
	"golang.org/x/crypto/hkdf"
)

var errInvalidPacket = errors.New("quic: invalid packet")

// headerProtectionSampleSize is the size of the ciphertext sample used for header protection.
// https://www.rfc-editor.org/rfc/rfc9001#section-5.4.2
const headerProtectionSampleSize = 16

// aeadOverhead is the difference in size between the AEAD output and input.
// All cipher suites defined for use with QUIC have 16 bytes of overhead.
const aeadOverhead = 16

// A headerKey applies or removes header protection.
// https://www.rfc-editor.org/rfc/rfc9001#section-5.4
type headerKey struct {
	hp headerProtection
}

func (k headerKey) isSet() bool {
	return k.hp != nil
}

func (k *headerKey) init(suite uint16, secret []byte) {
	h, keySize := hashForSuite(suite)
	hpKey := hkdfExpandLabel(h.New, secret, "quic hp", nil, keySize)
	switch suite {
	case tls.TLS_AES_128_GCM_SHA256, tls.TLS_AES_256_GCM_SHA384:
		c, err := aes.NewCipher(hpKey)
		if err != nil {
			panic(err)
		}
		k.hp = &aesHeaderProtection{cipher: c}
	case tls.TLS_CHACHA20_POLY1305_SHA256:
		k.hp = chaCha20HeaderProtection{hpKey}
	default:
		panic("BUG: unknown cipher suite")
	}
}

// protect applies header protection.
// pnumOff is the offset of the packet number in the packet.
func (k headerKey) protect(hdr []byte, pnumOff int) {
	// Apply header protection.
	pnumSize := int(hdr[0]&0x03) + 1
	sample := hdr[pnumOff+4:][:headerProtectionSampleSize]
	mask := k.hp.headerProtection(sample)
	if isLongHeader(hdr[0]) {
		hdr[0] ^= mask[0] & 0x0f
	} else {
		hdr[0] ^= mask[0] & 0x1f
	}
	for i := 0; i < pnumSize; i++ {
		hdr[pnumOff+i] ^= mask[1+i]
	}
}

// unprotect removes header protection.
// pnumOff is the offset of the packet number in the packet.
// pnumMax is the largest packet number seen in the number space of this packet.
func (k headerKey) unprotect(pkt []byte, pnumOff int, pnumMax packetNumber) (hdr, pay []byte, pnum packetNumber, _ error) {
	if len(pkt) < pnumOff+4+headerProtectionSampleSize {
		return nil, nil, 0, errInvalidPacket
	}
	numpay := pkt[pnumOff:]
	sample := numpay[4:][:headerProtectionSampleSize]
	mask := k.hp.headerProtection(sample)
	if isLongHeader(pkt[0]) {
		pkt[0] ^= mask[0] & 0x0f
	} else {
		pkt[0] ^= mask[0] & 0x1f
	}
	pnumLen := int(pkt[0]&0x03) + 1
	pnum = packetNumber(0)
	for i := 0; i < pnumLen; i++ {
		numpay[i] ^= mask[1+i]
		pnum = (pnum << 8) | packetNumber(numpay[i])
	}
	pnum = decodePacketNumber(pnumMax, pnum, pnumLen)
	hdr = pkt[:pnumOff+pnumLen]
	pay = numpay[pnumLen:]
	return hdr, pay, pnum, nil
}

// headerProtection is the  header_protection function as defined in:
// https://www.rfc-editor.org/rfc/rfc9001#section-5.4.1
//
// This function takes a sample of the packet ciphertext
// and returns a 5-byte mask which will be applied to the
// protected portions of the packet header.
type headerProtection interface {
	headerProtection(sample []byte) (mask [5]byte)
}

// AES-based header protection.
// https://www.rfc-editor.org/rfc/rfc9001#section-5.4.3
type aesHeaderProtection struct {
	cipher  cipher.Block
	scratch [aes.BlockSize]byte
}

func (hp *aesHeaderProtection) headerProtection(sample []byte) (mask [5]byte) {
	hp.cipher.Encrypt(hp.scratch[:], sample)
	copy(mask[:], hp.scratch[:])
	return mask
}

// ChaCha20-based header protection.
// https://www.rfc-editor.org/rfc/rfc9001#section-5.4.4
type chaCha20HeaderProtection struct {
	key []byte
}

func (hp chaCha20HeaderProtection) headerProtection(sample []byte) (mask [5]byte) {
	counter := uint32(sample[3])<<24 | uint32(sample[2])<<16 | uint32(sample[1])<<8 | uint32(sample[0])
	nonce := sample[4:16]
	c, err := chacha20.NewUnauthenticatedCipher(hp.key, nonce)
	if err != nil {
		panic(err)
	}
	c.SetCounter(counter)
	c.XORKeyStream(mask[:], mask[:])
	return mask
}

// A packetKey applies or removes packet protection.
// https://www.rfc-editor.org/rfc/rfc9001#section-5.1
type packetKey struct {
	aead cipher.AEAD // AEAD function used for packet protection.
	iv   []byte      // IV used to construct the AEAD nonce.
}

func (k *packetKey) init(suite uint16, secret []byte) {
	// https://www.rfc-editor.org/rfc/rfc9001#section-5.1
	h, keySize := hashForSuite(suite)
	key := hkdfExpandLabel(h.New, secret, "quic key", nil, keySize)
	switch suite {
	case tls.TLS_AES_128_GCM_SHA256, tls.TLS_AES_256_GCM_SHA384:
		k.aead = newAESAEAD(key)
	case tls.TLS_CHACHA20_POLY1305_SHA256:
		k.aead = newChaCha20AEAD(key)
	default:
		panic("BUG: unknown cipher suite")
	}
	k.iv = hkdfExpandLabel(h.New, secret, "quic iv", nil, k.aead.NonceSize())
}

func newAESAEAD(key []byte) cipher.AEAD {
	c, err := aes.NewCipher(key)
	if err != nil {
		panic(err)
	}
	aead, err := cipher.NewGCM(c)
	if err != nil {
		panic(err)
	}
	return aead
}

func newChaCha20AEAD(key []byte) cipher.AEAD {
	var err error
	aead, err := chacha20poly1305.New(key)
	if err != nil {
		panic(err)
	}
	return aead
}

func (k packetKey) protect(hdr, pay []byte, pnum packetNumber) []byte {
	k.xorIV(pnum)
	defer k.xorIV(pnum)
	return k.aead.Seal(hdr, k.iv, pay, hdr)
}

func (k packetKey) unprotect(hdr, pay []byte, pnum packetNumber) (dec []byte, err error) {
	k.xorIV(pnum)
	defer k.xorIV(pnum)
	return k.aead.Open(pay[:0], k.iv, pay, hdr)
}

// xorIV xors the packet protection IV with the packet number.
func (k packetKey) xorIV(pnum packetNumber) {
	k.iv[len(k.iv)-8] ^= uint8(pnum >> 56)
	k.iv[len(k.iv)-7] ^= uint8(pnum >> 48)
	k.iv[len(k.iv)-6] ^= uint8(pnum >> 40)
	k.iv[len(k.iv)-5] ^= uint8(pnum >> 32)
	k.iv[len(k.iv)-4] ^= uint8(pnum >> 24)
	k.iv[len(k.iv)-3] ^= uint8(pnum >> 16)
	k.iv[len(k.iv)-2] ^= uint8(pnum >> 8)
	k.iv[len(k.iv)-1] ^= uint8(pnum)
}

// A fixedKeys is a header protection key and fixed packet protection key.
// The packet protection key is fixed (it does not update).
//
// Fixed keys are used for Initial and Handshake keys, which do not update.
type fixedKeys struct {
	hdr headerKey
	pkt packetKey
}

func (k *fixedKeys) init(suite uint16, secret []byte) {
	k.hdr.init(suite, secret)
	k.pkt.init(suite, secret)
}

func (k fixedKeys) isSet() bool {
	return k.hdr.hp != nil
}

// protect applies packet protection to a packet.
//
// On input, hdr contains the packet header, pay the unencrypted payload,
// pnumOff the offset of the packet number in the header, and pnum the untruncated
// packet number.
//
// protect returns the result of appending the encrypted payload to hdr and
// applying header protection.
func (k fixedKeys) protect(hdr, pay []byte, pnumOff int, pnum packetNumber) []byte {
	pkt := k.pkt.protect(hdr, pay, pnum)
	k.hdr.protect(pkt, pnumOff)
	return pkt
}

// unprotect removes packet protection from a packet.
//
// On input, pkt contains the full protected packet, pnumOff the offset of
// the packet number in the header, and pnumMax the largest packet number
// seen in the number space of this packet.
//
// unprotect removes header protection from the header in pkt, and returns
// the unprotected payload and packet number.
func (k fixedKeys) unprotect(pkt []byte, pnumOff int, pnumMax packetNumber) (pay []byte, num packetNumber, err error) {
	hdr, pay, pnum, err := k.hdr.unprotect(pkt, pnumOff, pnumMax)
	if err != nil {
		return nil, 0, err
	}
	pay, err = k.pkt.unprotect(hdr, pay, pnum)
	if err != nil {
		return nil, 0, err
	}
	return pay, pnum, nil
}

// A fixedKeyPair is a read/write pair of fixed keys.
type fixedKeyPair struct {
	r, w fixedKeys
}

func (k *fixedKeyPair) discard() {
	*k = fixedKeyPair{}
}

func (k *fixedKeyPair) canRead() bool {
	return k.r.isSet()
}

func (k *fixedKeyPair) canWrite() bool {
	return k.w.isSet()
}

// An updatingKeys is a header protection key and updatable packet protection key.
// updatingKeys are used for 1-RTT keys, where the packet protection key changes
// over the lifetime of a connection.
// https://www.rfc-editor.org/rfc/rfc9001#section-6
type updatingKeys struct {
	suite      uint16
	hdr        headerKey
	pkt        [2]packetKey // current, next
	nextSecret []byte       // secret used to generate pkt[1]
}

func (k *updatingKeys) init(suite uint16, secret []byte) {
	k.suite = suite
	k.hdr.init(suite, secret)
	// Initialize pkt[1] with secret_0, and then call update to generate secret_1.
	k.pkt[1].init(suite, secret)
	k.nextSecret = secret
	k.update()
}

// update performs a key update.
// The current key in pkt[0] is discarded.
// The next key in pkt[1] becomes the current key.
// A new next key is generated in pkt[1].
func (k *updatingKeys) update() {
	k.nextSecret = updateSecret(k.suite, k.nextSecret)
	k.pkt[0] = k.pkt[1]
	k.pkt[1].init(k.suite, k.nextSecret)
}

func updateSecret(suite uint16, secret []byte) (nextSecret []byte) {
	h, _ := hashForSuite(suite)
	return hkdfExpandLabel(h.New, secret, "quic ku", nil, len(secret))
}

// An updatingKeyPair is a read/write pair of updating keys.
//
// We keep two keys (current and next) in both read and write directions.
// When an incoming packet's phase matches the current phase bit,
// we unprotect it using the current keys; otherwise we use the next keys.
//
// When updating=false, outgoing packets are protected using the current phase.
//
// An update is initiated and updating is set to true when:
//   - we decide to initiate a key update; or
//   - we successfully unprotect a packet using the next keys,
//     indicating the peer has initiated a key update.
//
// When updating=true, outgoing packets are protected using the next phase.
// We do not change the current phase bit or generate new keys yet.
//
// The update concludes when we receive an ACK frame for a packet sent
// with the next keys. At this time, we set updating to false, flip the
// phase bit, and update the keys. This permits us to handle up to 1-RTT
// of reordered packets before discarding the previous phase's keys after
// an update.
type updatingKeyPair struct {
	phase        uint8 // current key phase (r.pkt[0], w.pkt[0])
	updating     bool
	authFailures int64        // total packet unprotect failures
	minSent      packetNumber // min packet number sent since entering the updating state
	minReceived  packetNumber // min packet number received in the next phase
	updateAfter  packetNumber // packet number after which to initiate key update
	r, w         updatingKeys
}

func (k *updatingKeyPair) init() {
	// 1-RTT packets until the first key update.
	//
	// We perform the first key update early in the connection so a peer
	// which does not support key updates will fail rapidly,
	// rather than after the connection has been long established.
	//
	// The QUIC interop runner "keyupdate" test requires that the client
	// initiate a key rotation early in the connection. Increasing this
	// value may cause interop test failures; if we do want to increase it,
	// we should either skip the keyupdate test or provide a way to override
	// the setting in interop tests.
	k.updateAfter = 100
}

func (k *updatingKeyPair) canRead() bool {
	return k.r.hdr.hp != nil
}

func (k *updatingKeyPair) canWrite() bool {
	return k.w.hdr.hp != nil
}

// handleAckFor finishes a key update after receiving an ACK for a packet in the next phase.
func (k *updatingKeyPair) handleAckFor(pnum packetNumber) {
	if k.updating && pnum >= k.minSent {
		k.updating = false
		k.phase ^= keyPhaseBit
		k.r.update()
		k.w.update()
	}
}

// needAckEliciting reports whether we should send an ack-eliciting packet in the next phase.
// The first packet sent in a phase is ack-eliciting, since the peer must acknowledge a
// packet in the new phase for us to finish the update.
func (k *updatingKeyPair) needAckEliciting() bool {
	return k.updating && k.minSent == maxPacketNumber
}

// protect applies packet protection to a packet.
// Parameters and returns are as for fixedKeyPair.protect.
func (k *updatingKeyPair) protect(hdr, pay []byte, pnumOff int, pnum packetNumber) []byte {
	var pkt []byte
	if k.updating {
		hdr[0] |= k.phase ^ keyPhaseBit
		pkt = k.w.pkt[1].protect(hdr, pay, pnum)
		k.minSent = min(pnum, k.minSent)
	} else {
		hdr[0] |= k.phase
		pkt = k.w.pkt[0].protect(hdr, pay, pnum)
		if pnum >= k.updateAfter {
			// Initiate a key update, starting with the next packet we send.
			//
			// We do this after protecting the current packet
			// to allow Conn.appendFrames to ensure that the first packet sent
			// in the new phase is ack-eliciting.
			k.updating = true
			k.minSent = maxPacketNumber
			k.minReceived = maxPacketNumber
			// The lowest confidentiality limit for a supported AEAD is 2^23 packets.
			// https://www.rfc-editor.org/rfc/rfc9001#section-6.6-5
			//
			// Schedule our next update for half that.
			k.updateAfter += (1 << 22)
		}
	}
	k.w.hdr.protect(pkt, pnumOff)
	return pkt
}

// unprotect removes packet protection from a packet.
// Parameters and returns are as for fixedKeyPair.unprotect.
func (k *updatingKeyPair) unprotect(pkt []byte, pnumOff int, pnumMax packetNumber) (pay []byte, pnum packetNumber, err error) {
	hdr, pay, pnum, err := k.r.hdr.unprotect(pkt, pnumOff, pnumMax)
	if err != nil {
		return nil, 0, err
	}
	// To avoid timing signals that might indicate the key phase bit is invalid,
	// we always attempt to unprotect the packet with one key.
	//
	// If the key phase bit matches and the packet number doesn't come after
	// the start of an in-progress update, use the current phase.
	// Otherwise, use the next phase.
	if hdr[0]&keyPhaseBit == k.phase && (!k.updating || pnum < k.minReceived) {
		pay, err = k.r.pkt[0].unprotect(hdr, pay, pnum)
	} else {
		pay, err = k.r.pkt[1].unprotect(hdr, pay, pnum)
		if err == nil {
			if !k.updating {
				// The peer has initiated a key update.
				k.updating = true
				k.minSent = maxPacketNumber
				k.minReceived = pnum
			} else {
				k.minReceived = min(pnum, k.minReceived)
			}
		}
	}
	if err != nil {
		k.authFailures++
		if k.authFailures >= aeadIntegrityLimit(k.r.suite) {
			return nil, 0, localTransportError{code: errAEADLimitReached}
		}
		return nil, 0, err
	}
	return pay, pnum, nil
}

// aeadIntegrityLimit returns the integrity limit for an AEAD:
// The maximum number of received packets that may fail authentication
// before closing the connection.
//
// https://www.rfc-editor.org/rfc/rfc9001#section-6.6-4
func aeadIntegrityLimit(suite uint16) int64 {
	switch suite {
	case tls.TLS_AES_128_GCM_SHA256, tls.TLS_AES_256_GCM_SHA384:
		return 1 << 52
	case tls.TLS_CHACHA20_POLY1305_SHA256:
		return 1 << 36
	default:
		panic("BUG: unknown cipher suite")
	}
}

// https://www.rfc-editor.org/rfc/rfc9001#section-5.2-2
var initialSalt = []byte{0x38, 0x76, 0x2c, 0xf7, 0xf5, 0x59, 0x34, 0xb3, 0x4d, 0x17, 0x9a, 0xe6, 0xa4, 0xc8, 0x0c, 0xad, 0xcc, 0xbb, 0x7f, 0x0a}

// initialKeys returns the keys used to protect Initial packets.
//
// The Initial packet keys are derived from the Destination Connection ID
// field in the client's first Initial packet.
//
// https://www.rfc-editor.org/rfc/rfc9001#section-5.2
func initialKeys(cid []byte, side connSide) fixedKeyPair {
	initialSecret := hkdf.Extract(sha256.New, cid, initialSalt)
	var clientKeys fixedKeys
	clientSecret := hkdfExpandLabel(sha256.New, initialSecret, "client in", nil, sha256.Size)
	clientKeys.init(tls.TLS_AES_128_GCM_SHA256, clientSecret)
	var serverKeys fixedKeys
	serverSecret := hkdfExpandLabel(sha256.New, initialSecret, "server in", nil, sha256.Size)
	serverKeys.init(tls.TLS_AES_128_GCM_SHA256, serverSecret)
	if side == clientSide {
		return fixedKeyPair{r: serverKeys, w: clientKeys}
	} else {
		return fixedKeyPair{w: serverKeys, r: clientKeys}
	}
}

// checkCipherSuite returns an error if suite is not a supported cipher suite.
func checkCipherSuite(suite uint16) error {
	switch suite {
	case tls.TLS_AES_128_GCM_SHA256:
	case tls.TLS_AES_256_GCM_SHA384:
	case tls.TLS_CHACHA20_POLY1305_SHA256:
	default:
		return errors.New("invalid cipher suite")
	}
	return nil
}

func hashForSuite(suite uint16) (h crypto.Hash, keySize int) {
	switch suite {
	case tls.TLS_AES_128_GCM_SHA256:
		return crypto.SHA256, 128 / 8
	case tls.TLS_AES_256_GCM_SHA384:
		return crypto.SHA384, 256 / 8
	case tls.TLS_CHACHA20_POLY1305_SHA256:
		return crypto.SHA256, chacha20.KeySize
	default:
		panic("BUG: unknown cipher suite")
	}
}

// hkdfExpandLabel implements HKDF-Expand-Label from RFC 8446, Section 7.1.
//
// Copied from crypto/tls/key_schedule.go.
func hkdfExpandLabel(hash func() hash.Hash, secret []byte, label string, context []byte, length int) []byte {
	var hkdfLabel cryptobyte.Builder
	hkdfLabel.AddUint16(uint16(length))
	hkdfLabel.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes([]byte("tls13 "))
		b.AddBytes([]byte(label))
	})
	hkdfLabel.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(context)
	})
	out := make([]byte, length)
	n, err := hkdf.Expand(hash, secret, hkdfLabel.BytesOrPanic()).Read(out)
	if err != nil || n != length {
		panic("quic: HKDF-Expand-Label invocation failed unexpectedly")
	}
	return out
}
