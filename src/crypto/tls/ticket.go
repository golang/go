// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"golang.org/x/crypto/cryptobyte"
	"io"
)

// sessionState contains the information that is serialized into a session
// ticket in order to later resume a connection.
type sessionState struct {
	vers         uint16
	cipherSuite  uint16
	masterSecret []byte
	certificates [][]byte
	// usedOldKey is true if the ticket from which this session came from
	// was encrypted with an older key and thus should be refreshed.
	usedOldKey bool
}

func (s *sessionState) marshal() []byte {
	length := 2 + 2 + 2 + len(s.masterSecret) + 2
	for _, cert := range s.certificates {
		length += 4 + len(cert)
	}

	ret := make([]byte, length)
	x := ret
	x[0] = byte(s.vers >> 8)
	x[1] = byte(s.vers)
	x[2] = byte(s.cipherSuite >> 8)
	x[3] = byte(s.cipherSuite)
	x[4] = byte(len(s.masterSecret) >> 8)
	x[5] = byte(len(s.masterSecret))
	x = x[6:]
	copy(x, s.masterSecret)
	x = x[len(s.masterSecret):]

	x[0] = byte(len(s.certificates) >> 8)
	x[1] = byte(len(s.certificates))
	x = x[2:]

	for _, cert := range s.certificates {
		x[0] = byte(len(cert) >> 24)
		x[1] = byte(len(cert) >> 16)
		x[2] = byte(len(cert) >> 8)
		x[3] = byte(len(cert))
		copy(x[4:], cert)
		x = x[4+len(cert):]
	}

	return ret
}

func (s *sessionState) unmarshal(data []byte) bool {
	if len(data) < 8 {
		return false
	}

	s.vers = uint16(data[0])<<8 | uint16(data[1])
	s.cipherSuite = uint16(data[2])<<8 | uint16(data[3])
	masterSecretLen := int(data[4])<<8 | int(data[5])
	data = data[6:]
	if len(data) < masterSecretLen {
		return false
	}

	s.masterSecret = data[:masterSecretLen]
	data = data[masterSecretLen:]

	if len(data) < 2 {
		return false
	}

	numCerts := int(data[0])<<8 | int(data[1])
	data = data[2:]

	s.certificates = make([][]byte, numCerts)
	for i := range s.certificates {
		if len(data) < 4 {
			return false
		}
		certLen := int(data[0])<<24 | int(data[1])<<16 | int(data[2])<<8 | int(data[3])
		data = data[4:]
		if certLen < 0 {
			return false
		}
		if len(data) < certLen {
			return false
		}
		s.certificates[i] = data[:certLen]
		data = data[certLen:]
	}

	return len(data) == 0
}

// sessionStateTLS13 is the content of a TLS 1.3 session ticket. Its first
// version (revision = 0) doesn't carry any of the information needed for 0-RTT
// validation and the nonce is always empty.
type sessionStateTLS13 struct {
	// uint8 version  = 0x0304;
	// uint8 revision = 0;
	cipherSuite      uint16
	createdAt        uint64
	resumptionSecret []byte      // opaque resumption_master_secret<1..2^8-1>;
	certificate      Certificate // CertificateEntry certificate_list<0..2^24-1>;
}

func (m *sessionStateTLS13) marshal() []byte {
	var b cryptobyte.Builder
	b.AddUint16(VersionTLS13)
	b.AddUint8(0) // revision
	b.AddUint16(m.cipherSuite)
	addUint64(&b, m.createdAt)
	b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(m.resumptionSecret)
	})
	marshalCertificate(&b, m.certificate)
	return b.BytesOrPanic()
}

func (m *sessionStateTLS13) unmarshal(data []byte) bool {
	*m = sessionStateTLS13{}
	s := cryptobyte.String(data)
	var version uint16
	var revision uint8
	return s.ReadUint16(&version) &&
		version == VersionTLS13 &&
		s.ReadUint8(&revision) &&
		revision == 0 &&
		s.ReadUint16(&m.cipherSuite) &&
		readUint64(&s, &m.createdAt) &&
		readUint8LengthPrefixed(&s, &m.resumptionSecret) &&
		len(m.resumptionSecret) != 0 &&
		unmarshalCertificate(&s, &m.certificate) &&
		s.Empty()
}

func (c *Conn) encryptTicket(state []byte) ([]byte, error) {
	encrypted := make([]byte, ticketKeyNameLen+aes.BlockSize+len(state)+sha256.Size)
	keyName := encrypted[:ticketKeyNameLen]
	iv := encrypted[ticketKeyNameLen : ticketKeyNameLen+aes.BlockSize]
	macBytes := encrypted[len(encrypted)-sha256.Size:]

	if _, err := io.ReadFull(c.config.rand(), iv); err != nil {
		return nil, err
	}
	key := c.config.ticketKeys()[0]
	copy(keyName, key.keyName[:])
	block, err := aes.NewCipher(key.aesKey[:])
	if err != nil {
		return nil, errors.New("tls: failed to create cipher while encrypting ticket: " + err.Error())
	}
	cipher.NewCTR(block, iv).XORKeyStream(encrypted[ticketKeyNameLen+aes.BlockSize:], state)

	mac := hmac.New(sha256.New, key.hmacKey[:])
	mac.Write(encrypted[:len(encrypted)-sha256.Size])
	mac.Sum(macBytes[:0])

	return encrypted, nil
}

func (c *Conn) decryptTicket(encrypted []byte) (plaintext []byte, usedOldKey bool) {
	if len(encrypted) < ticketKeyNameLen+aes.BlockSize+sha256.Size {
		return nil, false
	}

	keyName := encrypted[:ticketKeyNameLen]
	iv := encrypted[ticketKeyNameLen : ticketKeyNameLen+aes.BlockSize]
	macBytes := encrypted[len(encrypted)-sha256.Size:]
	ciphertext := encrypted[ticketKeyNameLen+aes.BlockSize : len(encrypted)-sha256.Size]

	keys := c.config.ticketKeys()
	keyIndex := -1
	for i, candidateKey := range keys {
		if bytes.Equal(keyName, candidateKey.keyName[:]) {
			keyIndex = i
			break
		}
	}

	if keyIndex == -1 {
		return nil, false
	}
	key := &keys[keyIndex]

	mac := hmac.New(sha256.New, key.hmacKey[:])
	mac.Write(encrypted[:len(encrypted)-sha256.Size])
	expected := mac.Sum(nil)

	if subtle.ConstantTimeCompare(macBytes, expected) != 1 {
		return nil, false
	}

	block, err := aes.NewCipher(key.aesKey[:])
	if err != nil {
		return nil, false
	}
	plaintext = make([]byte, len(ciphertext))
	cipher.NewCTR(block, iv).XORKeyStream(plaintext, ciphertext)

	return plaintext, keyIndex > 0
}
