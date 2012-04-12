// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto"
	"crypto/rsa"
	"crypto/subtle"
	"crypto/x509"
	"errors"
	"io"
)

func (c *Conn) serverHandshake() error {
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

	finishedHash := newFinishedHash(vers)
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
FindCipherSuite:
	for _, id := range clientHello.cipherSuites {
		for _, supported := range config.cipherSuites() {
			if id == supported {
				var candidate *cipherSuite

				for _, s := range cipherSuites {
					if s.id == id {
						candidate = s
						break
					}
				}
				if candidate == nil {
					continue
				}
				// Don't select a ciphersuite which we can't
				// support for this client.
				if candidate.elliptic && !ellipticOk {
					continue
				}
				suite = candidate
				break FindCipherSuite
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
	hello.cipherSuite = suite.id
	t := uint32(config.time().Unix())
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

	if len(config.Certificates) == 0 {
		return c.sendAlert(alertInternalError)
	}
	cert := &config.Certificates[0]
	if len(clientHello.serverName) > 0 {
		c.serverName = clientHello.serverName
		cert = config.getCertificateForName(clientHello.serverName)
	}

	if clientHello.ocspStapling && len(cert.OCSPStaple) > 0 {
		hello.ocspStapling = true
	}

	finishedHash.Write(hello.marshal())
	c.writeRecord(recordTypeHandshake, hello.marshal())

	certMsg := new(certificateMsg)
	certMsg.certificates = cert.Certificate
	finishedHash.Write(certMsg.marshal())
	c.writeRecord(recordTypeHandshake, certMsg.marshal())

	if hello.ocspStapling {
		certStatus := new(certificateStatusMsg)
		certStatus.statusType = statusTypeOCSP
		certStatus.response = cert.OCSPStaple
		finishedHash.Write(certStatus.marshal())
		c.writeRecord(recordTypeHandshake, certStatus.marshal())
	}

	keyAgreement := suite.ka()
	skx, err := keyAgreement.generateServerKeyExchange(config, cert, clientHello, hello)
	if err != nil {
		c.sendAlert(alertHandshakeFailure)
		return err
	}
	if skx != nil {
		finishedHash.Write(skx.marshal())
		c.writeRecord(recordTypeHandshake, skx.marshal())
	}

	if config.ClientAuth >= RequestClientCert {
		// Request a client certificate
		certReq := new(certificateRequestMsg)
		certReq.certificateTypes = []byte{certTypeRSASign}

		// An empty list of certificateAuthorities signals to
		// the client that it may send any certificate in response
		// to our request. When we know the CAs we trust, then
		// we can send them down, so that the client can choose
		// an appropriate certificate to give to us.
		if config.ClientCAs != nil {
			certReq.certificateAuthorities = config.ClientCAs.Subjects()
		}
		finishedHash.Write(certReq.marshal())
		c.writeRecord(recordTypeHandshake, certReq.marshal())
	}

	helloDone := new(serverHelloDoneMsg)
	finishedHash.Write(helloDone.marshal())
	c.writeRecord(recordTypeHandshake, helloDone.marshal())

	var pub *rsa.PublicKey // public key for client auth, if any

	msg, err = c.readHandshake()
	if err != nil {
		return err
	}

	// If we requested a client certificate, then the client must send a
	// certificate message, even if it's empty.
	if config.ClientAuth >= RequestClientCert {
		if certMsg, ok = msg.(*certificateMsg); !ok {
			return c.sendAlert(alertHandshakeFailure)
		}
		finishedHash.Write(certMsg.marshal())

		if len(certMsg.certificates) == 0 {
			// The client didn't actually send a certificate
			switch config.ClientAuth {
			case RequireAnyClientCert, RequireAndVerifyClientCert:
				c.sendAlert(alertBadCertificate)
				return errors.New("tls: client didn't provide a certificate")
			}
		}

		certs := make([]*x509.Certificate, len(certMsg.certificates))
		for i, asn1Data := range certMsg.certificates {
			if certs[i], err = x509.ParseCertificate(asn1Data); err != nil {
				c.sendAlert(alertBadCertificate)
				return errors.New("tls: failed to parse client certificate: " + err.Error())
			}
		}

		if c.config.ClientAuth >= VerifyClientCertIfGiven && len(certs) > 0 {
			opts := x509.VerifyOptions{
				Roots:         c.config.ClientCAs,
				CurrentTime:   c.config.time(),
				Intermediates: x509.NewCertPool(),
			}

			for i, cert := range certs {
				if i == 0 {
					continue
				}
				opts.Intermediates.AddCert(cert)
			}

			chains, err := certs[0].Verify(opts)
			if err != nil {
				c.sendAlert(alertBadCertificate)
				return errors.New("tls: failed to verify client's certificate: " + err.Error())
			}

			ok := false
			for _, ku := range certs[0].ExtKeyUsage {
				if ku == x509.ExtKeyUsageClientAuth {
					ok = true
					break
				}
			}
			if !ok {
				c.sendAlert(alertHandshakeFailure)
				return errors.New("tls: client's certificate's extended key usage doesn't permit it to be used for client authentication")
			}

			c.verifiedChains = chains
		}

		if len(certs) > 0 {
			if pub, ok = certs[0].PublicKey.(*rsa.PublicKey); !ok {
				return c.sendAlert(alertUnsupportedCertificate)
			}
			c.peerCertificates = certs
		}

		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
	}

	// Get client key exchange
	ckx, ok := msg.(*clientKeyExchangeMsg)
	if !ok {
		return c.sendAlert(alertUnexpectedMessage)
	}
	finishedHash.Write(ckx.marshal())

	// If we received a client cert in response to our certificate request message,
	// the client will send us a certificateVerifyMsg immediately after the
	// clientKeyExchangeMsg.  This message is a MD5SHA1 digest of all preceding
	// handshake-layer messages that is signed using the private key corresponding
	// to the client's certificate. This allows us to verify that the client is in
	// possession of the private key of the certificate.
	if len(c.peerCertificates) > 0 {
		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
		certVerify, ok := msg.(*certificateVerifyMsg)
		if !ok {
			return c.sendAlert(alertUnexpectedMessage)
		}

		digest := make([]byte, 0, 36)
		digest = finishedHash.serverMD5.Sum(digest)
		digest = finishedHash.serverSHA1.Sum(digest)
		err = rsa.VerifyPKCS1v15(pub, crypto.MD5SHA1, digest, certVerify.signature)
		if err != nil {
			c.sendAlert(alertBadCertificate)
			return errors.New("could not validate signature of connection nonces: " + err.Error())
		}

		finishedHash.Write(certVerify.marshal())
	}

	preMasterSecret, err := keyAgreement.processClientKeyExchange(config, cert, ckx, c.vers)
	if err != nil {
		c.sendAlert(alertHandshakeFailure)
		return err
	}

	masterSecret, clientMAC, serverMAC, clientKey, serverKey, clientIV, serverIV :=
		keysFromPreMasterSecret(c.vers, preMasterSecret, clientHello.random, hello.random, suite.macLen, suite.keyLen, suite.ivLen)

	clientCipher := suite.cipher(clientKey, clientIV, true /* for reading */)
	clientHash := suite.mac(c.vers, clientMAC)
	c.in.prepareCipherSpec(c.vers, clientCipher, clientHash)
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

	serverCipher := suite.cipher(serverKey, serverIV, false /* not for reading */)
	serverHash := suite.mac(c.vers, serverMAC)
	c.out.prepareCipherSpec(c.vers, serverCipher, serverHash)
	c.writeRecord(recordTypeChangeCipherSpec, []byte{1})

	finished := new(finishedMsg)
	finished.verifyData = finishedHash.serverSum(masterSecret)
	c.writeRecord(recordTypeHandshake, finished.marshal())

	c.handshakeComplete = true
	c.cipherSuite = suite.id

	return nil
}
