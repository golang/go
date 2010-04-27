// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/hmac"
	"crypto/rc4"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/subtle"
	"crypto/x509"
	"io"
	"os"
)

func (c *Conn) clientHandshake() os.Error {
	finishedHash := newFinishedHash()

	config := defaultConfig()

	hello := &clientHelloMsg{
		vers:               maxVersion,
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
		compressionMethods: []uint8{compressionNone},
		random:             make([]byte, 32),
	}

	t := uint32(config.Time())
	hello.random[0] = byte(t >> 24)
	hello.random[1] = byte(t >> 16)
	hello.random[2] = byte(t >> 8)
	hello.random[3] = byte(t)
	_, err := io.ReadFull(config.Rand, hello.random[4:])
	if err != nil {
		return c.sendAlert(alertInternalError)
	}

	finishedHash.Write(hello.marshal())
	c.writeRecord(recordTypeHandshake, hello.marshal())

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}
	serverHello, ok := msg.(*serverHelloMsg)
	if !ok {
		return c.sendAlert(alertUnexpectedMessage)
	}
	finishedHash.Write(serverHello.marshal())

	vers, ok := mutualVersion(serverHello.vers)
	if !ok {
		c.sendAlert(alertProtocolVersion)
	}
	c.vers = vers
	c.haveVers = true

	if serverHello.cipherSuite != TLS_RSA_WITH_RC4_128_SHA ||
		serverHello.compressionMethod != compressionNone {
		return c.sendAlert(alertUnexpectedMessage)
	}

	msg, err = c.readHandshake()
	if err != nil {
		return err
	}
	certMsg, ok := msg.(*certificateMsg)
	if !ok || len(certMsg.certificates) == 0 {
		return c.sendAlert(alertUnexpectedMessage)
	}
	finishedHash.Write(certMsg.marshal())

	certs := make([]*x509.Certificate, len(certMsg.certificates))
	for i, asn1Data := range certMsg.certificates {
		cert, err := x509.ParseCertificate(asn1Data)
		if err != nil {
			return c.sendAlert(alertBadCertificate)
		}
		certs[i] = cert
	}

	// TODO(agl): do better validation of certs: max path length, name restrictions etc.
	for i := 1; i < len(certs); i++ {
		if err := certs[i-1].CheckSignatureFrom(certs[i]); err != nil {
			return c.sendAlert(alertBadCertificate)
		}
	}

	// TODO(rsc): Find certificates for OS X 10.6.
	if false && config.RootCAs != nil {
		root := config.RootCAs.FindParent(certs[len(certs)-1])
		if root == nil {
			return c.sendAlert(alertBadCertificate)
		}
		if certs[len(certs)-1].CheckSignatureFrom(root) != nil {
			return c.sendAlert(alertBadCertificate)
		}
	}

	pub, ok := certs[0].PublicKey.(*rsa.PublicKey)
	if !ok {
		return c.sendAlert(alertUnsupportedCertificate)
	}

	msg, err = c.readHandshake()
	if err != nil {
		return err
	}
	shd, ok := msg.(*serverHelloDoneMsg)
	if !ok {
		return c.sendAlert(alertUnexpectedMessage)
	}
	finishedHash.Write(shd.marshal())

	ckx := new(clientKeyExchangeMsg)
	preMasterSecret := make([]byte, 48)
	preMasterSecret[0] = byte(hello.vers >> 8)
	preMasterSecret[1] = byte(hello.vers)
	_, err = io.ReadFull(config.Rand, preMasterSecret[2:])
	if err != nil {
		return c.sendAlert(alertInternalError)
	}

	ckx.ciphertext, err = rsa.EncryptPKCS1v15(config.Rand, pub, preMasterSecret)
	if err != nil {
		return c.sendAlert(alertInternalError)
	}

	finishedHash.Write(ckx.marshal())
	c.writeRecord(recordTypeHandshake, ckx.marshal())

	suite := cipherSuites[0]
	masterSecret, clientMAC, serverMAC, clientKey, serverKey :=
		keysFromPreMasterSecret11(preMasterSecret, hello.random, serverHello.random, suite.hashLength, suite.cipherKeyLength)

	cipher, _ := rc4.NewCipher(clientKey)

	c.out.prepareCipherSpec(cipher, hmac.New(sha1.New(), clientMAC))
	c.writeRecord(recordTypeChangeCipherSpec, []byte{1})

	finished := new(finishedMsg)
	finished.verifyData = finishedHash.clientSum(masterSecret)
	finishedHash.Write(finished.marshal())
	c.writeRecord(recordTypeHandshake, finished.marshal())

	cipher2, _ := rc4.NewCipher(serverKey)
	c.in.prepareCipherSpec(cipher2, hmac.New(sha1.New(), serverMAC))
	c.readRecord(recordTypeChangeCipherSpec)
	if c.err != nil {
		return c.err
	}

	msg, err = c.readHandshake()
	if err != nil {
		return err
	}
	serverFinished, ok := msg.(*finishedMsg)
	if !ok {
		return c.sendAlert(alertUnexpectedMessage)
	}

	verify := finishedHash.serverSum(masterSecret)
	if len(verify) != len(serverFinished.verifyData) ||
		subtle.ConstantTimeCompare(verify, serverFinished.verifyData) != 1 {
		return c.sendAlert(alertHandshakeFailure)
	}

	c.handshakeComplete = true
	c.cipherSuite = TLS_RSA_WITH_RC4_128_SHA
	return nil
}
