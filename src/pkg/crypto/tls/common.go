// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/rsa"
	"io"
	"os"
)

const (
	// maxTLSCiphertext is the maximum length of a plaintext payload.
	maxTLSPlaintext = 16384
	// maxTLSCiphertext is the maximum length payload after compression and encryption.
	maxTLSCiphertext = 16384 + 2048
	// maxHandshakeMsg is the largest single handshake message that we'll buffer.
	maxHandshakeMsg = 65536
	// defaultMajor and defaultMinor are the maximum TLS version that we support.
	defaultMajor = 3
	defaultMinor = 2
)


// TLS record types.
type recordType uint8

const (
	recordTypeChangeCipherSpec recordType = 20
	recordTypeAlert            recordType = 21
	recordTypeHandshake        recordType = 22
	recordTypeApplicationData  recordType = 23
)

// TLS handshake message types.
const (
	typeClientHello       uint8 = 1
	typeServerHello       uint8 = 2
	typeCertificate       uint8 = 11
	typeServerHelloDone   uint8 = 14
	typeClientKeyExchange uint8 = 16
	typeFinished          uint8 = 20
)

// TLS cipher suites.
var (
	TLS_RSA_WITH_RC4_128_SHA uint16 = 5
)

// TLS compression types.
var (
	compressionNone uint8 = 0
)

type ConnectionState struct {
	HandshakeComplete bool
	CipherSuite       string
	Error             alertType
}

// A Config structure is used to configure a TLS client or server. After one
// has been passed to a TLS function it must not be modified.
type Config struct {
	// Rand provides the source of entropy for nonces and RSA blinding.
	Rand io.Reader
	// Time returns the current time as the number of seconds since the epoch.
	Time         func() int64
	Certificates []Certificate
	RootCAs      *CASet
}

type Certificate struct {
	Certificate [][]byte
	PrivateKey  *rsa.PrivateKey
}

// A TLS record.
type record struct {
	contentType  recordType
	major, minor uint8
	payload      []byte
}

type handshakeMessage interface {
	marshal() []byte
}

type encryptor interface {
	// XORKeyStream xors the contents of the slice with bytes from the key stream.
	XORKeyStream(buf []byte)
}

// mutualVersion returns the protocol version to use given the advertised
// version of the peer.
func mutualVersion(theirMajor, theirMinor uint8) (major, minor uint8, ok bool) {
	// We don't deal with peers < TLS 1.0 (aka version 3.1).
	if theirMajor < 3 || theirMajor == 3 && theirMinor < 1 {
		return 0, 0, false
	}
	major = 3
	minor = 2
	if theirMinor < minor {
		minor = theirMinor
	}
	ok = true
	return
}

// A nop implements the NULL encryption and MAC algorithms.
type nop struct{}

func (nop) XORKeyStream(buf []byte) {}

func (nop) Write(buf []byte) (int, os.Error) { return len(buf), nil }

func (nop) Sum() []byte { return nil }

func (nop) Reset() {}

func (nop) Size() int { return 0 }
