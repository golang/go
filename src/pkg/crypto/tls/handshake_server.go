// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/rsa"
	"crypto/subtle"
	"crypto/x509"
	"io"
	"os"
)

func (c *Conn) serverHandshake() os.Error {
	config := c.config
	msg, err := c.readHandshake()
	if err != nil {
		return err
	}
	clientHello, ok := msg.(*clientHelloMsg)
	if !ok {
		return c.sendAlert(alertUnexpectedMessage)
	}
	vers, ok := mutualVersion(clientHello.vers)
	if !ok {
		return c.sendAlert(alertProtocolVersion)
	}
	c.vers = vers
	c.haveVers = true

	finishedHash := newFinishedHash()
	finishedHash.Write(clientHello.marshal())

	hello := new(serverHelloMsg)

	supportedCurve := false
Curves:
	for _, curve := range clientHello.supportedCurves {
		switch curve {
		case curveP256, curveP384, curveP521:
			supportedCurve = true
			break Curves
		}
	}

	supportedPointFormat := false
	for _, pointFormat := range clientHello.supportedPoints {
		if pointFormat == pointFormatUncompressed {
			supportedPointFormat = true
			break
		}
	}

	ellipticOk := supportedCurve && supportedPointFormat

	var suite *cipherSuite
	var suiteId uint16
	for _, id := range clientHello.cipherSuites {
		for _, supported := range config.cipherSuites() {
			if id == supported {
				suite = cipherSuites[id]
				// Don't select a ciphersuite which we can't
				// support for this client.
				if suite.elliptic && !ellipticOk {
					continue
				}
				suiteId = id
				break
			}
		}
	}

	foundCompression := false
	// We only support null compression, so check that the client offered it.
	for _, compression := range clientHello.compressionMethods {
		if compression == compressionNone {
			foundCompression = true
			break
		}
	}

	if suite == nil || !foundCompression {
		return c.sendAlert(alertHandshakeFailure)
	}

	hello.vers = vers
	hello.cipherSuite = suiteId
	t := uint32(config.time())
	hello.random = make([]byte, 32)
	hello.random[0] = byte(t >> 24)
	hello.random[1] = byte(t >> 16)
	hello.random[2] = byte(t >> 8)
	hello.random[3] = byte(t)
	_, err = io.ReadFull(config.rand(), hello.random[4:])
	if err != nil {
		return c.sendAlert(alertInternalError)
	}
	hello.compressionMethod = compressionNone
	if clientHello.nextProtoNeg {
		hello.nextProtoNeg = true
		hello.nextProtos = config.NextProtos
	}

	finishedHash.Write(hello.marshal())
	c.writeRecord(recordTypeHandshake, hello.marshal())

	if len(config.Certificates) == 0 {
		return c.sendAlert(alertInternalError)
	}

	certMsg := new(certificateMsg)
	certMsg.certificates = config.Certificates[0].Certificate
	finishedHash.Write(certMsg.marshal())
	c.writeRecord(recordTypeHandshake, certMsg.marshal())

	keyAgreement := suite.ka()

	skx, err := keyAgreement.generateServerKeyExchange(config, clientHello, hello)
	if err != nil {
		c.sendAlert(alertHandshakeFailure)
		return err
	}
	if skx != nil {
		finishedHash.Write(skx.marshal())
		c.writeRecord(recordTypeHandshake, skx.marshal())
	}

	if config.AuthenticateClient {
		// Request a client certificate
		certReq := new(certificateRequestMsg)
		certReq.certificateTypes = []byte{certTypeRSASign}
		// An empty list of certificateAuthorities signals to
		// the client that it may send any certificate in response
		// to our request.

		finishedHash.Write(certReq.marshal())
		c.writeRecord(recordTypeHandshake, certReq.marshal())
	}

	helloDone := new(serverHelloDoneMsg)
	finishedHash.Write(helloDone.marshal())
	c.writeRecord(recordTypeHandshake, helloDone.marshal())

	var pub *rsa.PublicKey
	if config.AuthenticateClient {
		// Get client certificate
		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
		certMsg, ok = msg.(*certificateMsg)
		if !ok {
			return c.sendAlert(alertUnexpectedMessage)
		}
		finishedHash.Write(certMsg.marshal())

		certs := make([]*x509.Certificate, len(certMsg.certificates))
		for i, asn1Data := range certMsg.certificates {
			cert, err := x509.ParseCertificate(asn1Data)
			if err != nil {
				c.sendAlert(alertBadCertificate)
				return os.ErrorString("could not parse client's certificate: " + err.String())
			}
			certs[i] = cert
		}

		// TODO(agl): do better validation of certs: max path length, name restrictions etc.
		for i := 1; i < len(certs); i++ {
			if err := certs[i-1].CheckSignatureFrom(certs[i]); err != nil {
				c.sendAlert(alertBadCertificate)
				return os.ErrorString("could not validate certificate signature: " + err.String())
			}
		}

		if len(certs) > 0 {
			key, ok := certs[0].PublicKey.(*rsa.PublicKey)
			if !ok {
				return c.sendAlert(alertUnsupportedCertificate)
			}
			pub = key
			c.peerCertificates = certs
		}
	}

	// Get client key exchange
	msg, err = c.readHandshake()
	if err != nil {
		return err
	}
	ckx, ok := msg.(*clientKeyExchangeMsg)
	if !ok {
		return c.sendAlert(alertUnexpectedMessage)
	}
	finishedHash.Write(ckx.marshal())

	// If we received a client cert in response to our certificate request message,
	// the client will send us a certificateVerifyMsg immediately after the
	// clientKeyExchangeMsg.  This message is a MD5SHA1 digest of all preceeding
	// handshake-layer messages that is signed using the private key corresponding
	// to the client's certificate. This allows us to verify that the client is in
	// posession of the private key of the certificate.
	if len(c.peerCertificates) > 0 {
		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
		certVerify, ok := msg.(*certificateVerifyMsg)
		if !ok {
			return c.sendAlert(alertUnexpectedMessage)
		}

		digest := make([]byte, 36)
		copy(digest[0:16], finishedHash.serverMD5.Sum())
		copy(digest[16:36], finishedHash.serverSHA1.Sum())
		err = rsa.VerifyPKCS1v15(pub, rsa.HashMD5SHA1, digest, certVerify.signature)
		if err != nil {
			c.sendAlert(alertBadCertificate)
			return os.ErrorString("could not validate signature of connection nonces: " + err.String())
		}

		finishedHash.Write(certVerify.marshal())
	}

	preMasterSecret, err := keyAgreement.processClientKeyExchange(config, ckx)
	if err != nil {
		c.sendAlert(alertHandshakeFailure)
		return err
	}

	masterSecret, clientMAC, serverMAC, clientKey, serverKey, clientIV, serverIV :=
		keysFromPreMasterSecret10(preMasterSecret, clientHello.random, hello.random, suite.macLen, suite.keyLen, suite.ivLen)

	clientCipher := suite.cipher(clientKey, clientIV, true /* for reading */ )
	clientHash := suite.mac(clientMAC)
	c.in.prepareCipherSpec(clientCipher, clientHash)
	c.readRecord(recordTypeChangeCipherSpec)
	if err := c.error(); err != nil {
		return err
	}

	if hello.nextProtoNeg {
		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
		nextProto, ok := msg.(*nextProtoMsg)
		if !ok {
			return c.sendAlert(alertUnexpectedMessage)
		}
		finishedHash.Write(nextProto.marshal())
		c.clientProtocol = nextProto.proto
	}

	msg, err = c.readHandshake()
	if err != nil {
		return err
	}
	clientFinished, ok := msg.(*finishedMsg)
	if !ok {
		return c.sendAlert(alertUnexpectedMessage)
	}

	verify := finishedHash.clientSum(masterSecret)
	if len(verify) != len(clientFinished.verifyData) ||
		subtle.ConstantTimeCompare(verify, clientFinished.verifyData) != 1 {
		return c.sendAlert(alertHandshakeFailure)
	}

	finishedHash.Write(clientFinished.marshal())

	serverCipher := suite.cipher(serverKey, serverIV, false /* not for reading */ )
	serverHash := suite.mac(serverMAC)
	c.out.prepareCipherSpec(serverCipher, serverHash)
	c.writeRecord(recordTypeChangeCipherSpec, []byte{1})

	finished := new(finishedMsg)
	finished.verifyData = finishedHash.serverSum(masterSecret)
	c.writeRecord(recordTypeHandshake, finished.marshal())

	c.handshakeComplete = true
	c.cipherSuite = suiteId

	return nil
}
