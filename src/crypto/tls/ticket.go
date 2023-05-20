// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"io"

	"golang.org/x/crypto/cryptobyte"
)

// A SessionState is a resumable session.
type SessionState struct {
	version uint16 // uint16 version;
	// uint8 revision = 1;
	cipherSuite uint16
	createdAt   uint64
	secret      []byte      // opaque master_secret<1..2^8-1>;
	certificate Certificate // CertificateEntry certificate_list<0..2^24-1>;
}

// Bytes encodes the session, including any private fields, so that it can be
// parsed by [ParseSessionState]. The encoding contains secret values.
//
// The specific encoding should be considered opaque and may change incompatibly
// between Go versions.
func (m *SessionState) Bytes() ([]byte, error) {
	var b cryptobyte.Builder
	b.AddUint16(m.version)
	b.AddUint8(1) // revision
	b.AddUint16(m.cipherSuite)
	addUint64(&b, m.createdAt)
	b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(m.secret)
	})
	marshalCertificate(&b, m.certificate)
	return b.Bytes()
}

// ParseSessionState parses a [SessionState] encoded by [SessionState.Bytes].
func ParseSessionState(data []byte) (*SessionState, error) {
	ss := &SessionState{}
	s := cryptobyte.String(data)
	var revision uint8
	if !s.ReadUint16(&ss.version) ||
		!s.ReadUint8(&revision) ||
		revision != 1 ||
		!s.ReadUint16(&ss.cipherSuite) ||
		!readUint64(&s, &ss.createdAt) ||
		!readUint8LengthPrefixed(&s, &ss.secret) ||
		len(ss.secret) == 0 ||
		!unmarshalCertificate(&s, &ss.certificate) ||
		!s.Empty() {
		return nil, errors.New("tls: invalid session encoding")
	}
	return ss, nil
}

func (c *Conn) encryptTicket(state []byte) ([]byte, error) {
	if len(c.ticketKeys) == 0 {
		return nil, errors.New("tls: internal error: session ticket keys unavailable")
	}

	encrypted := make([]byte, aes.BlockSize+len(state)+sha256.Size)
	iv := encrypted[:aes.BlockSize]
	ciphertext := encrypted[aes.BlockSize : len(encrypted)-sha256.Size]
	authenticated := encrypted[:len(encrypted)-sha256.Size]
	macBytes := encrypted[len(encrypted)-sha256.Size:]

	if _, err := io.ReadFull(c.config.rand(), iv); err != nil {
		return nil, err
	}
	key := c.ticketKeys[0]
	block, err := aes.NewCipher(key.aesKey[:])
	if err != nil {
		return nil, errors.New("tls: failed to create cipher while encrypting ticket: " + err.Error())
	}
	cipher.NewCTR(block, iv).XORKeyStream(ciphertext, state)

	mac := hmac.New(sha256.New, key.hmacKey[:])
	mac.Write(authenticated)
	mac.Sum(macBytes[:0])

	return encrypted, nil
}

func (c *Conn) decryptTicket(encrypted []byte) []byte {
	if len(encrypted) < aes.BlockSize+sha256.Size {
		return nil
	}

	iv := encrypted[:aes.BlockSize]
	ciphertext := encrypted[aes.BlockSize : len(encrypted)-sha256.Size]
	authenticated := encrypted[:len(encrypted)-sha256.Size]
	macBytes := encrypted[len(encrypted)-sha256.Size:]

	for _, key := range c.ticketKeys {
		mac := hmac.New(sha256.New, key.hmacKey[:])
		mac.Write(authenticated)
		expected := mac.Sum(nil)

		if subtle.ConstantTimeCompare(macBytes, expected) != 1 {
			continue
		}

		block, err := aes.NewCipher(key.aesKey[:])
		if err != nil {
			return nil
		}
		plaintext := make([]byte, len(ciphertext))
		cipher.NewCTR(block, iv).XORKeyStream(plaintext, ciphertext)

		return plaintext
	}

	return nil
}
