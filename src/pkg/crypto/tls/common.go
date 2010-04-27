// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/rand"
	"crypto/rsa"
	"io"
	"io/ioutil"
	"once"
	"time"
)

const (
	maxPlaintext    = 16384        // maximum plaintext payload length
	maxCiphertext   = 16384 + 2048 // maximum ciphertext payload length
	recordHeaderLen = 5            // record header length
	maxHandshake    = 65536        // maximum handshake we support (protocol max is 16 MB)

	minVersion = 0x0301 // minimum supported version - TLS 1.0
	maxVersion = 0x0302 // maximum supported version - TLS 1.1
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
	typeNextProtocol      uint8 = 67 // Not IANA assigned
)

// TLS cipher suites.
var (
	TLS_RSA_WITH_RC4_128_SHA uint16 = 5
)

// TLS compression types.
var (
	compressionNone uint8 = 0
)

// TLS extension numbers
var (
	extensionServerName   uint16 = 0
	extensionNextProtoNeg uint16 = 13172 // not IANA assigned
)

type ConnectionState struct {
	HandshakeComplete  bool
	CipherSuite        string
	Error              alert
	NegotiatedProtocol string
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
	// NextProtos is a list of supported, application level protocols.
	// Currently only server-side handling is supported.
	NextProtos []string
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
	unmarshal([]byte) bool
}

type encryptor interface {
	// XORKeyStream xors the contents of the slice with bytes from the key stream.
	XORKeyStream(buf []byte)
}

// mutualVersion returns the protocol version to use given the advertised
// version of the peer.
func mutualVersion(vers uint16) (uint16, bool) {
	if vers < minVersion {
		return 0, false
	}
	if vers > maxVersion {
		vers = maxVersion
	}
	return vers, true
}

// The defaultConfig is used in place of a nil *Config in the TLS server and client.
var varDefaultConfig *Config

func defaultConfig() *Config {
	once.Do(initDefaultConfig)
	return varDefaultConfig
}

// Possible certificate files; stop after finding one.
// On OS X we should really be using the Directory Services keychain
// but that requires a lot of Mach goo to get at.  Instead we use
// the same root set that curl uses.
var certFiles = []string{
	"/etc/ssl/certs/ca-certificates.crt", // Linux etc
	"/usr/share/curl/curl-ca-bundle.crt", // OS X
}

func initDefaultConfig() {
	roots := NewCASet()
	for _, file := range certFiles {
		data, err := ioutil.ReadFile(file)
		if err == nil {
			roots.SetFromPEM(data)
			break
		}
	}

	varDefaultConfig = &Config{
		Rand:    rand.Reader,
		Time:    time.Seconds,
		RootCAs: roots,
	}
}
