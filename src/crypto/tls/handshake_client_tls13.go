// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto"
	"crypto/hmac"
	"errors"
	"hash"
	"sync/atomic"
)

type clientHandshakeStateTLS13 struct {
	c             *Conn
	serverHello   *serverHelloMsg
	hello         *clientHelloMsg
	certReq       *certificateRequestMsgTLS13
	ecdheParams   ecdheParameters
	suite         *cipherSuiteTLS13
	transcript    hash.Hash
	masterSecret  []byte
	trafficSecret []byte // client_application_traffic_secret_0
	session       *ClientSessionState
}

func (hs *clientHandshakeStateTLS13) handshake() error {
	c := hs.c

	// The server must not select TLS 1.3 in a renegotiation. See RFC 8446,
	// sections 4.1.2 and 4.1.3.
	if c.handshakes > 0 {
		c.sendAlert(alertProtocolVersion)
		return errors.New("tls: server selected TLS 1.3 in a renegotiation")
	}

	// Consistency check on the presence of a keyShare and its parameters.
	if hs.ecdheParams == nil || len(hs.hello.keyShares) != 1 {
		return c.sendAlert(alertInternalError)
	}

	if err := hs.checkServerHelloOrHRR(); err != nil {
		return err
	}

	hs.transcript = hs.suite.hash.New()
	hs.transcript.Write(hs.hello.marshal())

	if bytes.Equal(hs.serverHello.random, helloRetryRequestRandom) {
		// The first ClientHello gets double-hashed into the transcript upon a
		// HelloRetryRequest. See RFC 8446, Section 4.4.1.
		chHash := hs.transcript.Sum(nil)
		hs.transcript.Reset()
		hs.transcript.Write([]byte{typeMessageHash, 0, 0, uint8(len(chHash))})
		hs.transcript.Write(chHash)
		hs.transcript.Write(hs.serverHello.marshal())

		if err := hs.processHelloRetryRequest(); err != nil {
			return err
		}

		hs.transcript.Write(hs.hello.marshal())
	}

	hs.transcript.Write(hs.serverHello.marshal())

	if err := hs.processServerHello(); err != nil {
		return err
	}
	if err := hs.establishHandshakeKeys(); err != nil {
		return err
	}
	if err := hs.readServerParameters(); err != nil {
		return err
	}
	if err := hs.doFullHandshake(); err != nil {
		return err
	}
	if err := hs.readServerFinished(); err != nil {
		return err
	}

	c.buffering = true
	if err := hs.sendClientCertificate(); err != nil {
		return err
	}
	if err := hs.sendClientFinished(); err != nil {
		return err
	}
	if _, err := c.flush(); err != nil {
		return err
	}

	atomic.StoreUint32(&c.handshakeStatus, 1)

	return nil
}

// checkServerHelloOrHRR does validity checks that apply to both ServerHello and
// HelloRetryRequest messages. It sets hs.suite.
func (hs *clientHandshakeStateTLS13) checkServerHelloOrHRR() error {
	c := hs.c

	if hs.serverHello.supportedVersion == 0 {
		c.sendAlert(alertMissingExtension)
		return errors.New("tls: server selected TLS 1.3 using the legacy version field")
	}

	if hs.serverHello.supportedVersion != VersionTLS13 {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server selected an invalid version after a HelloRetryRequest")
	}

	if hs.serverHello.vers != VersionTLS12 {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server sent an incorrect legacy version")
	}

	if hs.serverHello.nextProtoNeg ||
		len(hs.serverHello.nextProtos) != 0 ||
		hs.serverHello.ocspStapling ||
		hs.serverHello.ticketSupported ||
		hs.serverHello.secureRenegotiationSupported ||
		len(hs.serverHello.secureRenegotiation) != 0 ||
		len(hs.serverHello.alpnProtocol) != 0 ||
		len(hs.serverHello.scts) != 0 {
		c.sendAlert(alertUnsupportedExtension)
		return errors.New("tls: server sent a ServerHello extension forbidden in TLS 1.3")
	}

	if !bytes.Equal(hs.hello.sessionId, hs.serverHello.sessionId) {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server did not echo the legacy session ID")
	}

	if hs.serverHello.compressionMethod != compressionNone {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server selected unsupported compression format")
	}

	selectedSuite := mutualCipherSuiteTLS13(hs.hello.cipherSuites, hs.serverHello.cipherSuite)
	if hs.suite != nil && selectedSuite != hs.suite {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server changed cipher suite after a HelloRetryRequest")
	}
	if selectedSuite == nil {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server chose an unconfigured cipher suite")
	}
	hs.suite = selectedSuite
	c.cipherSuite = hs.suite.id

	return nil
}

// processHelloRetryRequest handles the HRR in hs.serverHello, modifies and
// resends hs.hello, and reads the new ServerHello into hs.serverHello.
func (hs *clientHandshakeStateTLS13) processHelloRetryRequest() error {
	c := hs.c

	if hs.serverHello.serverShare.group != 0 {
		c.sendAlert(alertDecodeError)
		return errors.New("tls: received malformed key_share extension")
	}

	curveID := hs.serverHello.selectedGroup
	if curveID == 0 {
		c.sendAlert(alertMissingExtension)
		return errors.New("tls: received HelloRetryRequest without selected group")
	}
	curveOK := false
	for _, id := range hs.hello.supportedCurves {
		if id == curveID {
			curveOK = true
			break
		}
	}
	if !curveOK {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server selected unsupported group")
	}
	if hs.ecdheParams.CurveID() == curveID {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server sent an unnecessary HelloRetryRequest message")
	}
	if _, ok := curveForCurveID(curveID); curveID != X25519 && !ok {
		c.sendAlert(alertInternalError)
		return errors.New("tls: CurvePreferences includes unsupported curve")
	}
	params, err := generateECDHEParameters(c.config.rand(), curveID)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}
	hs.ecdheParams = params
	hs.hello.keyShares = []keyShare{{group: curveID, data: params.PublicKey()}}

	hs.hello.cookie = hs.serverHello.cookie

	hs.hello.raw = nil
	if _, err := c.writeRecord(recordTypeHandshake, hs.hello.marshal()); err != nil {
		return err
	}

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	serverHello, ok := msg.(*serverHelloMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(serverHello, msg)
	}
	hs.serverHello = serverHello

	if err := hs.checkServerHelloOrHRR(); err != nil {
		return err
	}

	return nil
}

func (hs *clientHandshakeStateTLS13) processServerHello() error {
	c := hs.c

	if bytes.Equal(hs.serverHello.random, helloRetryRequestRandom) {
		c.sendAlert(alertUnexpectedMessage)
		return errors.New("tls: server sent two HelloRetryRequest messages")
	}

	if len(hs.serverHello.cookie) != 0 {
		c.sendAlert(alertUnsupportedExtension)
		return errors.New("tls: server sent a cookie in a normal ServerHello")
	}

	if hs.serverHello.selectedGroup != 0 {
		c.sendAlert(alertDecodeError)
		return errors.New("tls: malformed key_share extension")
	}

	if hs.serverHello.serverShare.group != hs.ecdheParams.CurveID() {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: server selected unsupported group")
	}

	return nil
}

func (hs *clientHandshakeStateTLS13) establishHandshakeKeys() error {
	c := hs.c

	sharedKey := hs.ecdheParams.SharedKey(hs.serverHello.serverShare.data)
	if sharedKey == nil {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: invalid server key share")
	}

	earlySecret := hs.suite.extract(nil, nil)
	handshakeSecret := hs.suite.extract(sharedKey,
		hs.suite.deriveSecret(earlySecret, "derived", nil))

	clientSecret := hs.suite.deriveSecret(handshakeSecret,
		clientHandshakeTrafficLabel, hs.transcript)
	c.out.setTrafficSecret(hs.suite, clientSecret)
	serverSecret := hs.suite.deriveSecret(handshakeSecret,
		serverHandshakeTrafficLabel, hs.transcript)
	c.in.setTrafficSecret(hs.suite, serverSecret)

	hs.masterSecret = hs.suite.extract(nil,
		hs.suite.deriveSecret(handshakeSecret, "derived", nil))

	return nil
}

func (hs *clientHandshakeStateTLS13) readServerParameters() error {
	c := hs.c

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	encryptedExtensions, ok := msg.(*encryptedExtensionsMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(encryptedExtensions, msg)
	}
	hs.transcript.Write(encryptedExtensions.marshal())

	if len(encryptedExtensions.alpnProtocol) != 0 && len(hs.hello.alpnProtocols) == 0 {
		c.sendAlert(alertUnsupportedExtension)
		return errors.New("tls: server advertised unrequested ALPN extension")
	}
	c.clientProtocol = encryptedExtensions.alpnProtocol

	return nil
}

func (hs *clientHandshakeStateTLS13) doFullHandshake() error {
	c := hs.c

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	certReq, ok := msg.(*certificateRequestMsgTLS13)
	if ok {
		hs.transcript.Write(certReq.marshal())

		hs.certReq = certReq

		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
	}

	certMsg, ok := msg.(*certificateMsgTLS13)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(certMsg, msg)
	}
	if len(certMsg.certificate.Certificate) == 0 {
		c.sendAlert(alertDecodeError)
		return errors.New("tls: received empty certificates message")
	}
	hs.transcript.Write(certMsg.marshal())

	c.scts = certMsg.certificate.SignedCertificateTimestamps
	c.ocspResponse = certMsg.certificate.OCSPStaple

	if err := c.verifyServerCertificate(certMsg.certificate.Certificate); err != nil {
		return err
	}

	msg, err = c.readHandshake()
	if err != nil {
		return err
	}

	certVerify, ok := msg.(*certificateVerifyMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(certVerify, msg)
	}

	// See RFC 8446, Section 4.4.3.
	if !isSupportedSignatureAlgorithm(certVerify.signatureAlgorithm, hs.hello.supportedSignatureAlgorithms) {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: invalid certificate signature algorithm")
	}
	sigType := signatureFromSignatureScheme(certVerify.signatureAlgorithm)
	sigHash, err := hashFromSignatureScheme(certVerify.signatureAlgorithm)
	if sigType == 0 || err != nil {
		c.sendAlert(alertInternalError)
		return err
	}
	if sigType == signaturePKCS1v15 || sigHash == crypto.SHA1 {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: invalid certificate signature algorithm")
	}
	h := sigHash.New()
	writeSignedMessage(h, serverSignatureContext, hs.transcript)
	if err := verifyHandshakeSignature(sigType, c.peerCertificates[0].PublicKey,
		sigHash, h.Sum(nil), certVerify.signature); err != nil {
		c.sendAlert(alertDecryptError)
		return errors.New("tls: invalid certificate signature")
	}

	hs.transcript.Write(certVerify.marshal())

	return nil
}

func (hs *clientHandshakeStateTLS13) readServerFinished() error {
	c := hs.c

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	finished, ok := msg.(*finishedMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(finished, msg)
	}

	// See RFC 8446, sections 4.4.4 and 4.4.
	finishedKey := hs.suite.expandLabel(c.in.trafficSecret, "finished", nil, hs.suite.hash.Size())
	expectedMAC := hmac.New(hs.suite.hash.New, finishedKey)
	expectedMAC.Write(hs.transcript.Sum(nil))
	if !hmac.Equal(expectedMAC.Sum(nil), finished.verifyData) {
		c.sendAlert(alertDecryptError)
		return errors.New("tls: invalid finished hash")
	}

	hs.transcript.Write(finished.marshal())

	// Derive secrets that take context through the server Finished.

	hs.trafficSecret = hs.suite.deriveSecret(hs.masterSecret,
		clientApplicationTrafficLabel, hs.transcript)
	serverSecret := hs.suite.deriveSecret(hs.masterSecret,
		serverApplicationTrafficLabel, hs.transcript)
	c.in.setTrafficSecret(hs.suite, serverSecret)

	c.ekm = hs.suite.exportKeyingMaterial(hs.masterSecret, hs.transcript)

	return nil
}

func (hs *clientHandshakeStateTLS13) sendClientCertificate() error {
	if hs.certReq == nil {
		return nil
	}

	return errors.New("tls: TLS 1.3 client authentication unimplemented") // TODO(filippo)
}

func (hs *clientHandshakeStateTLS13) sendClientFinished() error {
	c := hs.c

	finishedKey := hs.suite.expandLabel(c.out.trafficSecret, "finished", nil, hs.suite.hash.Size())
	verifyData := hmac.New(hs.suite.hash.New, finishedKey)
	verifyData.Write(hs.transcript.Sum(nil))
	finished := &finishedMsg{
		verifyData: verifyData.Sum(nil),
	}

	hs.transcript.Write(finished.marshal())
	if _, err := c.writeRecord(recordTypeHandshake, finished.marshal()); err != nil {
		return err
	}

	c.out.setTrafficSecret(hs.suite, hs.trafficSecret)

	return nil
}
