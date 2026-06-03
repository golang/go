// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/binary"
	"net/netip"
	"time"

	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/net/internal/quic/quicwire"
)

// AEAD and nonce used to compute the Retry Integrity Tag.
// https://www.rfc-editor.org/rfc/rfc9001#section-5.8
var (
	retrySecret = []byte{0xbe, 0x0c, 0x69, 0x0b, 0x9f, 0x66, 0x57, 0x5a, 0x1d, 0x76, 0x6b, 0x54, 0xe3, 0x68, 0xc8, 0x4e}
	retryNonce  = []byte{0x46, 0x15, 0x99, 0xd3, 0x5d, 0x63, 0x2b, 0xf2, 0x23, 0x98, 0x25, 0xbb}
	retryAEAD   = func() cipher.AEAD {
		c, err := aes.NewCipher(retrySecret)
		if err != nil {
			panic(err)
		}
		aead, err := cipher.NewGCM(c)
		if err != nil {
			panic(err)
		}
		return aead
	}()
)

// retryTokenValidityPeriod is how long we accept a Retry packet token after sending it.
const retryTokenValidityPeriod = 5 * time.Second

// retryState generates and validates an endpoint's retry tokens.
type retryState struct {
	aead cipher.AEAD
}

func (rs *retryState) init() error {
	// Retry tokens are authenticated using a per-server key chosen at start time.
	// TODO: Provide a way for the user to set this key.
	secret := make([]byte, chacha20poly1305.KeySize)
	if _, err := rand.Read(secret); err != nil {
		return err
	}
	aead, err := chacha20poly1305.NewX(secret)
	if err != nil {
		panic(err)
	}
	rs.aead = aead
	return nil
}

// Retry tokens are encrypted with an AEAD.
// The plaintext contains the time the token was created and
// the original destination connection ID.
// The additional data contains the sender's source address and original source connection ID.
// The token nonce is randomly generated.
// We use the nonce as the Source Connection ID of the Retry packet.
// Since the 24-byte XChaCha20-Poly1305 nonce is too large to fit in a 20-byte connection ID,
// we include the remaining 4 bytes of nonce in the token.
//
// Token {
//   Last 4 Bytes of Nonce (32),
//   Ciphertext (..),
// }
//
// Plaintext {
//   Timestamp (64),
//   Original Destination Connection ID,
// }
//
//
// Additional Data {
//   Original Source Connection ID Length (8),
//   Original Source Connection ID (..),
//   IP Address (32..128),
//   Port (16),
// }
//
// TODO: Consider using AES-256-GCM-SIV once crypto/tls supports it.

func (rs *retryState) makeToken(now time.Time, srcConnID, origDstConnID []byte, addr netip.AddrPort) (token, newDstConnID []byte, err error) {
	nonce := make([]byte, rs.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, nil, err
	}

	var plaintext []byte
	plaintext = binary.BigEndian.AppendUint64(plaintext, uint64(now.Unix()))
	plaintext = append(plaintext, origDstConnID...)

	token = append(token, nonce[maxConnIDLen:]...)
	token = rs.aead.Seal(token, nonce, plaintext, rs.additionalData(srcConnID, addr))
	return token, nonce[:maxConnIDLen], nil
}

func (rs *retryState) validateToken(now time.Time, token, srcConnID, dstConnID []byte, addr netip.AddrPort) (origDstConnID []byte, ok bool) {
	tokenNonceLen := rs.aead.NonceSize() - maxConnIDLen
	if len(token) < tokenNonceLen {
		return nil, false
	}
	nonce := append([]byte{}, dstConnID...)
	nonce = append(nonce, token[:tokenNonceLen]...)
	ciphertext := token[tokenNonceLen:]
	if len(nonce) != rs.aead.NonceSize() {
		return nil, false
	}

	plaintext, err := rs.aead.Open(nil, nonce, ciphertext, rs.additionalData(srcConnID, addr))
	if err != nil {
		return nil, false
	}
	if len(plaintext) < 8 {
		return nil, false
	}
	when := time.Unix(int64(binary.BigEndian.Uint64(plaintext)), 0)
	origDstConnID = plaintext[8:]

	// We allow for tokens created in the future (up to the validity period),
	// which likely indicates that the system clock was adjusted backwards.
	if d := abs(now.Sub(when)); d > retryTokenValidityPeriod {
		return nil, false
	}

	return origDstConnID, true
}

func (rs *retryState) additionalData(srcConnID []byte, addr netip.AddrPort) []byte {
	var additional []byte
	additional = quicwire.AppendUint8Bytes(additional, srcConnID)
	additional = append(additional, addr.Addr().AsSlice()...)
	additional = binary.BigEndian.AppendUint16(additional, addr.Port())
	return additional
}

func (e *Endpoint) validateInitialAddress(now time.Time, p genericLongPacket, peerAddr netip.AddrPort) (origDstConnID []byte, ok bool) {
	// The retry token is at the start of an Initial packet's data.
	token, n := quicwire.ConsumeUint8Bytes(p.data)
	if n < 0 {
		// We've already validated that the packet is at least 1200 bytes long,
		// so there's no way for even a maximum size token to not fit.
		// Check anyway.
		return nil, false
	}
	if len(token) == 0 {
		// The sender has not provided a token.
		// Send a Retry packet to them with one.
		e.sendRetry(now, p, peerAddr)
		return nil, false
	}
	origDstConnID, ok = e.retry.validateToken(now, token, p.srcConnID, p.dstConnID, peerAddr)
	if !ok {
		// This does not seem to be a valid token.
		// Close the connection with an INVALID_TOKEN error.
		// https://www.rfc-editor.org/rfc/rfc9000#section-8.1.2-5
		e.sendConnectionClose(p, peerAddr, errInvalidToken)
		return nil, false
	}
	return origDstConnID, true
}

func (e *Endpoint) sendRetry(now time.Time, p genericLongPacket, peerAddr netip.AddrPort) {
	token, srcConnID, err := e.retry.makeToken(now, p.srcConnID, p.dstConnID, peerAddr)
	if err != nil {
		return
	}
	b := encodeRetryPacket(p.dstConnID, retryPacket{
		dstConnID: p.srcConnID,
		srcConnID: srcConnID,
		token:     token,
	})
	e.sendDatagram(datagram{
		b:        b,
		peerAddr: peerAddr,
	})
}

type retryPacket struct {
	dstConnID []byte
	srcConnID []byte
	token     []byte
}

func encodeRetryPacket(originalDstConnID []byte, p retryPacket) []byte {
	// Retry packets include an integrity tag, computed by AEAD_AES_128_GCM over
	// the original destination connection ID followed by the Retry packet
	// (less the integrity tag itself).
	// https://www.rfc-editor.org/rfc/rfc9001#section-5.8
	//
	// Create the pseudo-packet (including the original DCID), append the tag,
	// and return the Retry packet.
	var b []byte
	b = quicwire.AppendUint8Bytes(b, originalDstConnID) // Original Destination Connection ID
	start := len(b)                                     // start of the Retry packet
	b = append(b, headerFormLong|fixedBit|longPacketTypeRetry)
	b = binary.BigEndian.AppendUint32(b, quicVersion1) // Version
	b = quicwire.AppendUint8Bytes(b, p.dstConnID)      // Destination Connection ID
	b = quicwire.AppendUint8Bytes(b, p.srcConnID)      // Source Connection ID
	b = append(b, p.token...)                          // Token
	b = retryAEAD.Seal(b, retryNonce, nil, b)          // Retry Integrity Tag
	return b[start:]
}

func parseRetryPacket(b, origDstConnID []byte) (p retryPacket, ok bool) {
	const retryIntegrityTagLength = 128 / 8

	lp, ok := parseGenericLongHeaderPacket(b)
	if !ok {
		return retryPacket{}, false
	}
	if len(lp.data) < retryIntegrityTagLength {
		return retryPacket{}, false
	}
	gotTag := lp.data[len(lp.data)-retryIntegrityTagLength:]

	// Create the pseudo-packet consisting of the original destination connection ID
	// followed by the Retry packet (less the integrity tag).
	// Use this to validate the packet integrity tag.
	pseudo := quicwire.AppendUint8Bytes(nil, origDstConnID)
	pseudo = append(pseudo, b[:len(b)-retryIntegrityTagLength]...)
	wantTag := retryAEAD.Seal(nil, retryNonce, nil, pseudo)
	if !bytes.Equal(gotTag, wantTag) {
		return retryPacket{}, false
	}

	token := lp.data[:len(lp.data)-retryIntegrityTagLength]
	return retryPacket{
		dstConnID: lp.dstConnID,
		srcConnID: lp.srcConnID,
		token:     token,
	}, true
}
