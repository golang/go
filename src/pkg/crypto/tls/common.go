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

// ConnectionState records basic TLS details about the connection.
type ConnectionState struct {
	HandshakeComplete  bool
	CipherSuite        uint16
	NegotiatedProtocol string
}

// A Config structure is used to configure a TLS client or server. After one
// has been passed to a TLS function it must not be modified.
type Config struct {
	// Rand provides the source of entropy for nonces and RSA blinding.
	// If Rand is nil, TLS uses the cryptographic random reader in package
	// crypto/rand.
	Rand io.Reader

	// Time returns the current time as the number of seconds since the epoch.
	// If Time is nil, TLS uses the system time.Seconds.
	Time func() int64

	// Certificates contains one or more certificate chains
	// to present to the other side of the connection.
	// Server configurations must include at least one certificate.
	Certificates []Certificate

	// RootCAs defines the set of root certificate authorities
	// that clients use when verifying server certificates.
	// If RootCAs is nil, TLS uses the host's root CA set.
	RootCAs *CASet

	// NextProtos is a list of supported, application level protocols.
	// Currently only server-side handling is supported.
	NextProtos []string

	// ServerName is included in the client's handshake to support virtual
	// hosting.
	ServerName string

	// AuthenticateClient controls whether a server will request a certificate
	// from the client. It does not require that the client send a
	// certificate nor does it require that the certificate sent be
	// anything more than self-signed.
	AuthenticateClient bool
}

func (c *Config) rand() io.Reader {
	r := c.Rand
	if r == nil {
		return rand.Reader
	}
	return r
}

func (c *Config) time() int64 {
	t := c.Time
	if t == nil {
		t = time.Seconds
	}
	return t()
}

func (c *Config) rootCAs() *CASet {
	s := c.RootCAs
	if s == nil {
		s = defaultRoots()
	}
	return s
}

// A Certificate is a chain of one or more certificates, leaf first.
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

var emptyConfig Config

func defaultConfig() *Config {
	return &emptyConfig
}

// Possible certificate files; stop after finding one.
// On OS X we should really be using the Directory Services keychain
// but that requires a lot of Mach goo to get at.  Instead we use
// the same root set that curl uses.
var certFiles = []string{
	"/etc/ssl/certs/ca-certificates.crt", // Linux etc
	"/usr/share/curl/curl-ca-bundle.crt", // OS X
}

var once sync.Once

func defaultRoots() *CASet {
	once.Do(initDefaultRoots)
	return varDefaultRoots
}

var varDefaultRoots *CASet

func initDefaultRoots() {
	roots := NewCASet()
	for _, file := range certFiles {
		data, err := ioutil.ReadFile(file)
		if err == nil {
			roots.SetFromPEM(data)
			break
		}
	}
	varDefaultRoots = roots
}
