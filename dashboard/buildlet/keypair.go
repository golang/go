// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

package buildlet

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net"
	"time"
)

// KeyPair is the TLS public certificate PEM file and its associated
// private key PEM file that a builder will use for its HTTPS
// server. The zero value means no HTTPs, which is used by the
// coordinator for machines running within a firewall.
type KeyPair struct {
	CertPEM string
	KeyPEM  string
}

func (kp KeyPair) IsZero() bool { return kp == KeyPair{} }

// Password returns the SHA1 of the KeyPEM. This is used as the HTTP
// Basic Auth password.
func (kp KeyPair) Password() string {
	if kp.KeyPEM != "" {
		return fmt.Sprintf("%x", sha1.Sum([]byte(kp.KeyPEM)))
	}
	return ""
}

// tlsDialer returns a TLS dialer for http.Transport.DialTLS that expects
// exactly our TLS cert.
func (kp KeyPair) tlsDialer() func(network, addr string) (net.Conn, error) {
	if kp.IsZero() {
		// Unused.
		return nil
	}
	wantCert, _ := tls.X509KeyPair([]byte(kp.CertPEM), []byte(kp.KeyPEM))
	var wantPubKey *rsa.PublicKey = &wantCert.PrivateKey.(*rsa.PrivateKey).PublicKey

	return func(network, addr string) (net.Conn, error) {
		if network != "tcp" {
			return nil, fmt.Errorf("unexpected network %q", network)
		}
		plainConn, err := net.Dial("tcp", addr)
		if err != nil {
			return nil, err
		}
		tlsConn := tls.Client(plainConn, &tls.Config{InsecureSkipVerify: true})
		if err := tlsConn.Handshake(); err != nil {
			return nil, err
		}
		certs := tlsConn.ConnectionState().PeerCertificates
		if len(certs) < 1 {
			return nil, errors.New("no server peer certificate")
		}
		cert := certs[0]
		peerPubRSA, ok := cert.PublicKey.(*rsa.PublicKey)
		if !ok {
			return nil, fmt.Errorf("peer cert was a %T; expected RSA", cert.PublicKey)
		}
		if peerPubRSA.N.Cmp(wantPubKey.N) != 0 {
			return nil, fmt.Errorf("unexpected TLS certificate")
		}
		return tlsConn, nil
	}
}

// NoKeyPair is used by the coordinator to speak http directly to buildlets,
// inside their firewall, without TLS.
var NoKeyPair = KeyPair{}

func NewKeyPair() (KeyPair, error) {
	fail := func(err error) (KeyPair, error) { return KeyPair{}, err }
	failf := func(format string, args ...interface{}) (KeyPair, error) { return fail(fmt.Errorf(format, args...)) }

	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return failf("rsa.GenerateKey: %s", err)
	}

	notBefore := time.Now()
	notAfter := notBefore.Add(5 * 365 * 24 * time.Hour) // 5 years

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return failf("failed to generate serial number: %s", err)
	}

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"Gopher Co"},
		},
		NotBefore: notBefore,
		NotAfter:  notAfter,

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              []string{"localhost"},
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return failf("Failed to create certificate: %s", err)
	}

	var certOut bytes.Buffer
	pem.Encode(&certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	var keyOut bytes.Buffer
	pem.Encode(&keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	return KeyPair{
		CertPEM: certOut.String(),
		KeyPEM:  keyOut.String(),
	}, nil
}
