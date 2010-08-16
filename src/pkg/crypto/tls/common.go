// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/rand"
	"crypto/rsa"
	"io"
	"io/ioutil"
	"sync"
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
	typeClientHello        uint8 = 1
	typeServerHello        uint8 = 2
	typeCertificate        uint8 = 11
	typeCertificateRequest uint8 = 13
	typeServerHelloDone    uint8 = 14
	typeCertificateVerify  uint8 = 15
	typeClientKeyExchange  uint8 = 16
	typeFinished           uint8 = 20
	typeCertificateStatus  uint8 = 22
	typeNextProtocol       uint8 = 67 // Not IANA assigned
)

// TLS cipher suites.
const (
	TLS_RSA_WITH_RC4_128_SHA uint16 = 5
)

// TLS compression types.
const (
	compressionNone uint8 = 0
)

// TLS extension numbers
var (
	extensionServerName    uint16 = 0
	extensionStatusRequest uint16 = 5
	extensionNextProtoNeg  uint16 = 13172 // not IANA assigned
)

// TLS CertificateStatusType (RFC 3546)
const (
	statusTypeOCSP uint8 = 1
)

// Certificate types (for certificateRequestMsg)
const (
	certTypeRSASign    = 1 // A certificate containing an RSA key
	certTypeDSSSign    = 2 // A certificate containing a DSA key
	certTypeRSAFixedDH = 3 // A certificate containing a static DH key
	certTypeDSSFixedDH = 4 // A certficiate containing a static DH key
	// Rest of these are reserved by the TLS spec
)

type ConnectionState struct {
	HandshakeComplete  bool
	CipherSuite        uint16
	NegotiatedProtocol string
}

// A Config structure is used to configure a TLS client or server. After one
// has been passed to a TLS function it must not be modified.
type Config struct {
	// Rand provides the source of entropy for nonces and RSA blinding.
	Rand io.Reader
	// Time returns the current time as the number of seconds since the epoch.
	Time func() int64
	// Certificates contains one or more certificate chains.
	Certificates []Certificate
	RootCAs      *CASet
	// NextProtos is a list of supported, application level protocols.
	// Currently only server-side handling is supported.
	NextProtos []string
	// ServerName is included in the client's handshake to support virtual
	// hosting.
	ServerName string
	// AuthenticateClient determines if a server will request a certificate
	// from the client. It does not require that the client send a
	// certificate nor, if it does, that the certificate is anything more
	// than self-signed.
	AuthenticateClient bool
}

type Certificate struct {
	// Certificate contains a chain of one or more certificates. Leaf
	// certificate first.
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

var once sync.Once

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
