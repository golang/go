// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"context"
	"crypto"
	"crypto/hmac"
	"crypto/rsa"
	"errors"
	"hash"
	"io"
	"reflect"
	"sync/atomic"
	"time"
	"unsafe"
)

// maxClientPSKIdentities is the number of client PSK identities the server will
// attempt to validate. It will ignore the rest not to let cheap ClientHello
// messages cause too much work In session ticket decryption attempts.
const maxClientPSKIdentities = 5

type serverHandshakeStateTLS13 struct {
	c               *Conn
	ctx             context.Context
	clientHello     *clientHelloMsg
	hello           *serverHelloMsg
	sentDummyCCS    bool
	usingPSK        bool
	suite           *cipherSuiteTLS13
	cert            *Certificate
	sigAlg          SignatureScheme
	earlySecret     []byte
	sharedKey       []byte
	handshakeSecret []byte
	masterSecret    []byte
	trafficSecret   []byte // client_application_traffic_secret_0
	transcript      hash.Hash
	clientFinished  []byte
}

func (hs *serverHandshakeStateTLS13) handshake() error {
	c := hs.c

	// For an overview of the TLS 1.3 handshake, see RFC 8446, Section 2.
	if err := hs.processClientHello(); err != nil {
		return err
	}
	if err := hs.checkForResumption(); err != nil {
		return err
	}
	if err := hs.pickCertificate(); err != nil {
		return err
	}
	c.Buffering = true
	if err := hs.sendServerParameters(); err != nil {
		return err
	}
	if err := hs.sendServerCertificate(); err != nil {
		return err
	}
	if err := hs.sendServerFinished(); err != nil {
		return err
	}
	// Note that at this point we could start sending application Data without
	// waiting for the client's second flight, but the application might not
	// expect the lack of replay protection of the ClientHello parameters.
	if _, err := c.flush(); err != nil {
		return err
	}
	if err := hs.readClientCertificate(); err != nil {
		return err
	}

	return nil
}

func (hs *serverHandshakeStateTLS13) handshakeSecondary() error {
	c := hs.c

	if err := hs.readClientFinished(); err != nil {
		return err
	}

	atomic.StoreUint32(&c.HandshakeStatus, 1)

	return nil
}

func (hs *serverHandshakeStateTLS13) processClientHello() error {
	c := hs.c

	hs.hello = new(serverHelloMsg)

	// TLS 1.3 froze the ServerHello.legacy_version field, and uses
	// supported_versions instead. See RFC 8446, sections 4.1.3 and 4.2.1.
	hs.hello.vers = VersionTLS12
	hs.hello.supportedVersion = c.Vers

	if len(hs.clientHello.SupportedVersions) == 0 {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: client used the legacy Version field to negotiate TLS 1.3")
	}

	// Abort if the client is doing a fallback and landing lower than what we
	// support. See RFC 7507, which however does not specify the interaction
	// with supported_versions. The only difference is that with
	// supported_versions a client has a chance to attempt a [TLS 1.2, TLS 1.4]
	// handshake In case TLS 1.3 is broken but 1.2 is not. Alas, In that case,
	// it will have to drop the TLS_FALLBACK_SCSV protection if it falls back to
	// TLS 1.2, because a TLS 1.3 server would abort here. The situation before
	// supported_versions was not better because there was just no way to do a
	// TLS 1.4 handshake without risking the server selecting TLS 1.3.
	for _, id := range hs.clientHello.CipherSuites {
		if id == TLS_FALLBACK_SCSV {
			// Use c.Vers instead of max(supported_versions) because an attacker
			// could defeat this by adding an arbitrary high Version otherwise.
			if c.Vers < c.config.maxSupportedVersion() {
				c.sendAlert(alertInappropriateFallback)
				return errors.New("tls: client using inappropriate protocol fallback")
			}
			break
		}
	}

	if len(hs.clientHello.CompressionMethods) != 1 ||
		hs.clientHello.CompressionMethods[0] != compressionNone {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: TLS 1.3 client supports illegal compression methods")
	}

	hs.hello.random = make([]byte, 32)
	if _, err := io.ReadFull(c.config.rand(), hs.hello.random); err != nil {
		c.sendAlert(alertInternalError)
		return err
	}

	if len(hs.clientHello.SecureRenegotiation) != 0 {
		c.sendAlert(alertHandshakeFailure)
		return errors.New("tls: initial handshake had non-empty renegotiation extension")
	}

	if hs.clientHello.EarlyData {
		// See RFC 8446, Section 4.2.10 for the complicated behavior required
		// here. The scenario is that a different server at our address offered
		// to accept early Data In the past, which we can't handle. For now, all
		// 0-RTT enabled session tickets need to expire before a Go server can
		// replace a server or join a pool. That's the same requirement that
		// applies to mixing or replacing with any TLS 1.2 server.
		c.sendAlert(alertUnsupportedExtension)
		return errors.New("tls: client sent unexpected early Data")
	}

	hs.hello.sessionId = hs.clientHello.SessionId
	hs.hello.compressionMethod = compressionNone

	preferenceList := defaultCipherSuitesTLS13
	if !hasAESGCMHardwareSupport || !aesgcmPreferred(hs.clientHello.CipherSuites) {
		preferenceList = defaultCipherSuitesTLS13NoAES
	}
	for _, suiteID := range preferenceList {
		hs.suite = mutualCipherSuiteTLS13(hs.clientHello.CipherSuites, suiteID)
		if hs.suite != nil {
			break
		}
	}
	if hs.suite == nil {
		c.sendAlert(alertHandshakeFailure)
		return errors.New("tls: no Cipher suite supported by both client and server")
	}
	c.CipherSuite = hs.suite.Id
	hs.hello.cipherSuite = hs.suite.Id
	hs.transcript = hs.suite.hash.New()

	// Pick the ECDHE Group In server preference order, but give priority to
	// groups with a key share, to avoid a HelloRetryRequest round-trip.
	var selectedGroup CurveID
	var clientKeyShare *keyShare
GroupSelection:
	for _, preferredGroup := range c.config.curvePreferences() {
		for _, ks := range hs.clientHello.KeyShares {
			if ks.Group == preferredGroup {
				selectedGroup = ks.Group
				clientKeyShare = &ks
				break GroupSelection
			}
		}
		if selectedGroup != 0 {
			continue
		}
		for _, group := range hs.clientHello.SupportedCurves {
			if group == preferredGroup {
				selectedGroup = group
				break
			}
		}
	}
	if selectedGroup == 0 {
		c.sendAlert(alertHandshakeFailure)
		return errors.New("tls: no ECDHE curve supported by both client and server")
	}
	if clientKeyShare == nil {
		if err := hs.doHelloRetryRequest(selectedGroup); err != nil {
			return err
		}
		clientKeyShare = &hs.clientHello.KeyShares[0]
	}

	if _, ok := curveForCurveID(selectedGroup); selectedGroup != X25519 && !ok {
		c.sendAlert(alertInternalError)
		return errors.New("tls: CurvePreferences includes unsupported curve")
	}
	params, err := generateECDHEParameters(c.config.rand(), selectedGroup)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}
	hs.hello.serverShare = keyShare{Group: selectedGroup, Data: params.PublicKey()}
	hs.sharedKey = params.SharedKey(clientKeyShare.Data)
	if hs.sharedKey == nil {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: invalid client key share")
	}

	c.ServerName = hs.clientHello.ServerName
	return nil
}

func (hs *serverHandshakeStateTLS13) checkForResumption() error {
	c := hs.c

	if c.config.SessionTicketsDisabled {
		return nil
	}

	modeOK := false
	for _, mode := range hs.clientHello.PskModes {
		if mode == pskModeDHE {
			modeOK = true
			break
		}
	}
	if !modeOK {
		return nil
	}

	if len(hs.clientHello.PskIdentities) != len(hs.clientHello.PskBinders) {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: invalid or missing PSK binders")
	}
	if len(hs.clientHello.PskIdentities) == 0 {
		return nil
	}

	for i, identity := range hs.clientHello.PskIdentities {
		if i >= maxClientPSKIdentities {
			break
		}

		plaintext, _ := c.decryptTicket(identity.Label)
		if plaintext == nil {
			continue
		}
		sessionState := new(sessionStateTLS13)
		if ok := sessionState.unmarshal(plaintext); !ok {
			continue
		}

		createdAt := time.Unix(int64(sessionState.createdAt), 0)
		if c.config.time().Sub(createdAt) > maxSessionTicketLifetime {
			continue
		}

		// We don't check the obfuscated ticket age because it's affected by
		// clock skew and it's only a freshness signal useful for shrinking the
		// window for replay attacks, which don't affect us as we don't do 0-RTT.

		pskSuite := cipherSuiteTLS13ByID(sessionState.cipherSuite)
		if pskSuite == nil || pskSuite.hash != hs.suite.hash {
			continue
		}

		// PSK connections don't re-establish client certificates, but carry
		// them over In the session ticket. Ensure the presence of client certs
		// In the ticket is consistent with the configured requirements.
		sessionHasClientCerts := len(sessionState.certificate.Certificate) != 0
		needClientCerts := requiresClientCert(c.config.ClientAuth)
		if needClientCerts && !sessionHasClientCerts {
			continue
		}
		if sessionHasClientCerts && c.config.ClientAuth == NoClientCert {
			continue
		}

		psk := hs.suite.expandLabel(sessionState.resumptionSecret, "resumption",
			nil, hs.suite.hash.Size())
		hs.earlySecret = hs.suite.extract(psk, nil)
		binderKey := hs.suite.deriveSecret(hs.earlySecret, resumptionBinderLabel, nil)
		// Clone the transcript In case a HelloRetryRequest was recorded.
		transcript := cloneHash(hs.transcript, hs.suite.hash)
		if transcript == nil {
			c.sendAlert(alertInternalError)
			return errors.New("tls: internal error: failed to clone hash")
		}
		transcript.Write(hs.clientHello.marshalWithoutBinders())
		pskBinder := hs.suite.finishedHash(binderKey, transcript)
		if !hmac.Equal(hs.clientHello.PskBinders[i], pskBinder) {
			c.sendAlert(alertDecryptError)
			return errors.New("tls: invalid PSK binder")
		}

		c.DidResume = true
		if err := c.processCertsFromClient(sessionState.certificate); err != nil {
			return err
		}

		hs.hello.selectedIdentityPresent = true
		hs.hello.selectedIdentity = uint16(i)
		hs.usingPSK = true
		return nil
	}

	return nil
}

// cloneHash uses the encoding.BinaryMarshaler and encoding.BinaryUnmarshaler
// interfaces implemented by standard library hashes to clone the state of In
// to a new instance of h. It returns nil if the operation fails.
func cloneHash(in hash.Hash, h crypto.Hash) hash.Hash {
	// Recreate the interface to avoid importing encoding.
	type binaryMarshaler interface {
		MarshalBinary() (data []byte, err error)
		UnmarshalBinary(data []byte) error
	}
	marshaler, ok := in.(binaryMarshaler)
	if !ok {
		return nil
	}
	state, err := marshaler.MarshalBinary()
	if err != nil {
		return nil
	}
	out := h.New()
	unmarshaler, ok := out.(binaryMarshaler)
	if !ok {
		return nil
	}
	if err := unmarshaler.UnmarshalBinary(state); err != nil {
		return nil
	}
	return out
}

func (hs *serverHandshakeStateTLS13) pickCertificate() error {
	c := hs.c

	// Only one of PSK and certificates are used at a time.
	if hs.usingPSK {
		return nil
	}

	// signature_algorithms is required In TLS 1.3. See RFC 8446, Section 4.2.3.
	if len(hs.clientHello.SupportedSignatureAlgorithms) == 0 {
		return c.sendAlert(alertMissingExtension)
	}

	certificate, err := c.config.getCertificate(clientHelloInfo(hs.ctx, c, hs.clientHello))
	if err != nil {
		if err == errNoCertificates {
			c.sendAlert(alertUnrecognizedName)
		} else {
			c.sendAlert(alertInternalError)
		}
		return err
	}
	hs.sigAlg, err = selectSignatureScheme(c.Vers, certificate, hs.clientHello.SupportedSignatureAlgorithms)
	if err != nil {
		// getCertificate returned a certificate that is unsupported or
		// incompatible with the client's signature algorithms.
		c.sendAlert(alertHandshakeFailure)
		return err
	}
	hs.cert = certificate

	return nil
}

// sendDummyChangeCipherSpec sends a ChangeCipherSpec record for compatibility
// with middleboxes that didn't implement TLS correctly. See RFC 8446, Appendix D.4.
func (hs *serverHandshakeStateTLS13) sendDummyChangeCipherSpec() error {
	if hs.sentDummyCCS {
		return nil
	}
	hs.sentDummyCCS = true

	_, err := hs.c.writeRecord(recordTypeChangeCipherSpec, []byte{1})
	return err
}

func (hs *serverHandshakeStateTLS13) doHelloRetryRequest(selectedGroup CurveID) error {
	c := hs.c

	// The first ClientHello gets double-hashed into the transcript upon a
	// HelloRetryRequest. See RFC 8446, Section 4.4.1.
	hs.transcript.Write(hs.clientHello.marshal())
	chHash := hs.transcript.Sum(nil)
	hs.transcript.Reset()
	hs.transcript.Write([]byte{typeMessageHash, 0, 0, uint8(len(chHash))})
	hs.transcript.Write(chHash)

	helloRetryRequest := &serverHelloMsg{
		vers:              hs.hello.vers,
		random:            helloRetryRequestRandom,
		sessionId:         hs.hello.sessionId,
		cipherSuite:       hs.hello.cipherSuite,
		compressionMethod: hs.hello.compressionMethod,
		supportedVersion:  hs.hello.supportedVersion,
		selectedGroup:     selectedGroup,
	}

	hs.transcript.Write(helloRetryRequest.marshal())
	if _, err := c.writeRecord(recordTypeHandshake, helloRetryRequest.marshal()); err != nil {
		return err
	}

	if err := hs.sendDummyChangeCipherSpec(); err != nil {
		return err
	}

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	clientHello, ok := msg.(*clientHelloMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(clientHello, msg)
	}

	if len(clientHello.KeyShares) != 1 || clientHello.KeyShares[0].Group != selectedGroup {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: client sent invalid key share In second ClientHello")
	}

	if clientHello.EarlyData {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: client indicated early Data In second ClientHello")
	}

	if illegalClientHelloChange(clientHello, hs.clientHello) {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: client illegally modified second ClientHello")
	}

	hs.clientHello = clientHello
	return nil
}

// illegalClientHelloChange reports whether the two ClientHello messages are
// different, with the exception of the changes allowed before and after a
// HelloRetryRequest. See RFC 8446, Section 4.1.2.
func illegalClientHelloChange(ch, ch1 *clientHelloMsg) bool {
	if len(ch.SupportedVersions) != len(ch1.SupportedVersions) ||
		len(ch.CipherSuites) != len(ch1.CipherSuites) ||
		len(ch.SupportedCurves) != len(ch1.SupportedCurves) ||
		len(ch.SupportedSignatureAlgorithms) != len(ch1.SupportedSignatureAlgorithms) ||
		len(ch.SupportedSignatureAlgorithmsCert) != len(ch1.SupportedSignatureAlgorithmsCert) ||
		len(ch.AlpnProtocols) != len(ch1.AlpnProtocols) {
		return true
	}
	for i := range ch.SupportedVersions {
		if ch.SupportedVersions[i] != ch1.SupportedVersions[i] {
			return true
		}
	}
	for i := range ch.CipherSuites {
		if ch.CipherSuites[i] != ch1.CipherSuites[i] {
			return true
		}
	}
	for i := range ch.SupportedCurves {
		if ch.SupportedCurves[i] != ch1.SupportedCurves[i] {
			return true
		}
	}
	for i := range ch.SupportedSignatureAlgorithms {
		if ch.SupportedSignatureAlgorithms[i] != ch1.SupportedSignatureAlgorithms[i] {
			return true
		}
	}
	for i := range ch.SupportedSignatureAlgorithmsCert {
		if ch.SupportedSignatureAlgorithmsCert[i] != ch1.SupportedSignatureAlgorithmsCert[i] {
			return true
		}
	}
	for i := range ch.AlpnProtocols {
		if ch.AlpnProtocols[i] != ch1.AlpnProtocols[i] {
			return true
		}
	}
	return ch.Vers != ch1.Vers ||
		!bytes.Equal(ch.Random, ch1.Random) ||
		!bytes.Equal(ch.SessionId, ch1.SessionId) ||
		!bytes.Equal(ch.CompressionMethods, ch1.CompressionMethods) ||
		ch.ServerName != ch1.ServerName ||
		ch.OcspStapling != ch1.OcspStapling ||
		!bytes.Equal(ch.SupportedPoints, ch1.SupportedPoints) ||
		ch.TicketSupported != ch1.TicketSupported ||
		!bytes.Equal(ch.SessionTicket, ch1.SessionTicket) ||
		ch.SecureRenegotiationSupported != ch1.SecureRenegotiationSupported ||
		!bytes.Equal(ch.SecureRenegotiation, ch1.SecureRenegotiation) ||
		ch.Scts != ch1.Scts ||
		!bytes.Equal(ch.Cookie, ch1.Cookie) ||
		!bytes.Equal(ch.PskModes, ch1.PskModes)
}

func (hs *serverHandshakeStateTLS13) sendServerParameters() error {
	c := hs.c

	hs.transcript.Write(hs.clientHello.marshal())
	hs.transcript.Write(hs.hello.marshal())
	if _, err := c.writeRecord(recordTypeHandshake, hs.hello.marshal()); err != nil {
		return err
	}

	if err := hs.sendDummyChangeCipherSpec(); err != nil {
		return err
	}

	earlySecret := hs.earlySecret
	if earlySecret == nil {
		earlySecret = hs.suite.extract(nil, nil)
	}
	hs.handshakeSecret = hs.suite.extract(hs.sharedKey,
		hs.suite.deriveSecret(earlySecret, "derived", nil))

	clientSecret := hs.suite.deriveSecret(hs.handshakeSecret,
		clientHandshakeTrafficLabel, hs.transcript)
	c.In.setTrafficSecret(hs.suite, clientSecret)
	serverSecret := hs.suite.deriveSecret(hs.handshakeSecret,
		serverHandshakeTrafficLabel, hs.transcript)
	c.Out.setTrafficSecret(hs.suite, serverSecret)

	err := c.config.writeKeyLog(keyLogLabelClientHandshake, hs.clientHello.Random, clientSecret)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}
	err = c.config.writeKeyLog(keyLogLabelServerHandshake, hs.clientHello.Random, serverSecret)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}

	encryptedExtensions := new(encryptedExtensionsMsg)

	selectedProto, err := negotiateALPN(c.config.NextProtos, hs.clientHello.AlpnProtocols)
	if err != nil {
		c.sendAlert(alertNoApplicationProtocol)
		return err
	}
	encryptedExtensions.alpnProtocol = selectedProto
	c.ClientProtocol = selectedProto

	hs.transcript.Write(encryptedExtensions.marshal())
	if _, err := c.writeRecord(recordTypeHandshake, encryptedExtensions.marshal()); err != nil {
		return err
	}

	return nil
}

func (hs *serverHandshakeStateTLS13) requestClientCert() bool {
	return hs.c.config.ClientAuth >= RequestClientCert && !hs.usingPSK
}

func (hs *serverHandshakeStateTLS13) sendServerCertificate() error {
	c := hs.c

	// Only one of PSK and certificates are used at a time.
	if hs.usingPSK {
		return nil
	}

	if hs.requestClientCert() {
		// Request a client certificate
		certReq := new(certificateRequestMsgTLS13)
		certReq.ocspStapling = true
		certReq.scts = true
		certReq.supportedSignatureAlgorithms = supportedSignatureAlgorithms
		if c.config.ClientCAs != nil {
			certReq.certificateAuthorities = c.config.ClientCAs.Subjects()
		}

		hs.transcript.Write(certReq.marshal())
		if _, err := c.writeRecord(recordTypeHandshake, certReq.marshal()); err != nil {
			return err
		}
	}

	certMsg := new(certificateMsgTLS13)

	certMsg.certificate = *hs.cert
	certMsg.scts = hs.clientHello.Scts && len(hs.cert.SignedCertificateTimestamps) > 0
	certMsg.ocspStapling = hs.clientHello.OcspStapling && len(hs.cert.OCSPStaple) > 0

	hs.transcript.Write(certMsg.marshal())
	if _, err := c.writeRecord(recordTypeHandshake, certMsg.marshal()); err != nil {
		return err
	}

	certVerifyMsg := new(certificateVerifyMsg)
	certVerifyMsg.hasSignatureAlgorithm = true
	certVerifyMsg.signatureAlgorithm = hs.sigAlg

	sigType, sigHash, err := typeAndHashFromSignatureScheme(hs.sigAlg)
	if err != nil {
		return c.sendAlert(alertInternalError)
	}

	signed := signedMessage(sigHash, serverSignatureContext, hs.transcript)
	signOpts := crypto.SignerOpts(sigHash)
	if sigType == signatureRSAPSS {
		signOpts = &rsa.PSSOptions{SaltLength: rsa.PSSSaltLengthEqualsHash, Hash: sigHash}
	}
	sig, err := hs.cert.PrivateKey.(crypto.Signer).Sign(c.config.rand(), signed, signOpts)
	if err != nil {
		public := hs.cert.PrivateKey.(crypto.Signer).Public()
		if rsaKey, ok := public.(*rsa.PublicKey); ok && sigType == signatureRSAPSS &&
			rsaKey.N.BitLen()/8 < sigHash.Size()*2+2 { // key too small for RSA-PSS
			c.sendAlert(alertHandshakeFailure)
		} else {
			c.sendAlert(alertInternalError)
		}
		return errors.New("tls: failed to sign handshake: " + err.Error())
	}
	certVerifyMsg.signature = sig

	hs.transcript.Write(certVerifyMsg.marshal())
	if _, err := c.writeRecord(recordTypeHandshake, certVerifyMsg.marshal()); err != nil {
		return err
	}

	return nil
}

func (hs *serverHandshakeStateTLS13) sendServerFinished() error {
	c := hs.c

	finished := &finishedMsg{
		verifyData: hs.suite.finishedHash(c.Out.TrafficSecret, hs.transcript),
	}

	hs.transcript.Write(finished.marshal())
	if _, err := c.writeRecord(recordTypeHandshake, finished.marshal()); err != nil {
		return err
	}

	// Derive secrets that take context through the server Finished.

	hs.masterSecret = hs.suite.extract(nil,
		hs.suite.deriveSecret(hs.handshakeSecret, "derived", nil))

	hs.trafficSecret = hs.suite.deriveSecret(hs.masterSecret,
		clientApplicationTrafficLabel, hs.transcript)
	serverSecret := hs.suite.deriveSecret(hs.masterSecret,
		serverApplicationTrafficLabel, hs.transcript)
	c.Out.setTrafficSecret(hs.suite, serverSecret)

	err := c.config.writeKeyLog(keyLogLabelClientTraffic, hs.clientHello.Random, hs.trafficSecret)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}
	err = c.config.writeKeyLog(keyLogLabelServerTraffic, hs.clientHello.Random, serverSecret)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}

	c.ekm = hs.suite.ExportKeyingMaterial(hs.masterSecret, hs.transcript)

	// If we did not request client certificates, at this point we can
	// precompute the client finished and roll the transcript forward to send
	// session tickets In our first flight.
	if !hs.requestClientCert() {
		if err := hs.sendSessionTickets(); err != nil {
			return err
		}
	}

	return nil
}

func (hs *serverHandshakeStateTLS13) shouldSendSessionTickets() bool {
	if hs.c.config.SessionTicketsDisabled {
		return false
	}

	// Don't send tickets the client wouldn't use. See RFC 8446, Section 4.2.9.
	for _, pskMode := range hs.clientHello.PskModes {
		if pskMode == pskModeDHE {
			return true
		}
	}
	return false
}

func (hs *serverHandshakeStateTLS13) sendSessionTickets() error {
	c := hs.c

	hs.clientFinished = hs.suite.finishedHash(c.In.TrafficSecret, hs.transcript)
	finishedMsg := &finishedMsg{
		verifyData: hs.clientFinished,
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Manual serialization //////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	c.HS.MasterSecret = hs.masterSecret
	c.HS.ClientFinished = hs.clientFinished
	c.HS.Suite = *hs.suite
	c.HS.TrafficSecret = hs.trafficSecret

	rs := reflect.ValueOf(hs.transcript).Elem()
	// hs.transcript.h field
	rf1 := rs.Field(0)
	rf1 = reflect.NewAt(rf1.Type(), unsafe.Pointer(rf1.UnsafeAddr())).Elem()
	rt1 := reflect.ValueOf(&c.HS.H).Elem()
	rtw1 := reflect.NewAt(rt1.Type(), unsafe.Pointer(rt1.UnsafeAddr())).Elem()
	rtw1.Set(rf1)

	// hs.transcript.x field
	rf2 := rs.Field(1)
	rf2 = reflect.NewAt(rf2.Type(), unsafe.Pointer(rf2.UnsafeAddr())).Elem()
	rt2 := reflect.ValueOf(&c.HS.X).Elem()
	rtw2 := reflect.NewAt(rt2.Type(), unsafe.Pointer(rt2.UnsafeAddr())).Elem()
	rtw2.Set(rf2)

	// hs.transcript.nx field
	rf3 := rs.Field(2)
	rf3 = reflect.NewAt(rf3.Type(), unsafe.Pointer(rf3.UnsafeAddr())).Elem()
	rt3 := reflect.ValueOf(&c.HS.Nx).Elem()
	rtw3 := reflect.NewAt(rt3.Type(), unsafe.Pointer(rt3.UnsafeAddr())).Elem()
	rtw3.Set(rf3)

	// hs.transcript.len field
	rf4 := rs.Field(3)
	rf4 = reflect.NewAt(rf4.Type(), unsafe.Pointer(rf4.UnsafeAddr())).Elem()
	rt4 := reflect.ValueOf(&c.HS.Len).Elem()
	rtw4 := reflect.NewAt(rt4.Type(), unsafe.Pointer(rt4.UnsafeAddr())).Elem()
	rtw4.Set(rf4)

	// hs.transcript.is224 field
	rf5 := rs.Field(4)
	rf5 = reflect.NewAt(rf5.Type(), unsafe.Pointer(rf5.UnsafeAddr())).Elem()
	rt5 := reflect.ValueOf(&c.HS.Is224).Elem()
	rtw5 := reflect.NewAt(rt5.Type(), unsafe.Pointer(rt5.UnsafeAddr())).Elem()
	rtw5.Set(rf5)

	// Serialize the In halfconn
	rsInHalfConnCipher := reflect.ValueOf(c.In.Cipher).Elem()
	rf1InHalfConnCipher := rsInHalfConnCipher.Field(0)
	rf1InHalfConnCipher = reflect.NewAt(rf1InHalfConnCipher.Type(), unsafe.Pointer(rf1InHalfConnCipher.UnsafeAddr())).Elem()
	rt1InHalfConnCipher := reflect.ValueOf(&c.InHalfConn.NonceMask).Elem()
	rtw1InHalfConnCipher := reflect.NewAt(rt1InHalfConnCipher.Type(), unsafe.Pointer(rt1InHalfConnCipher.UnsafeAddr())).Elem()
	rtw1InHalfConnCipher.Set(rf1InHalfConnCipher)

	switch rsInHalfConnCipher.Field(1).Elem().Elem().Type().Name() {
	case "gcmAsm":
		{
			rf2InHalfConnCipher := rsInHalfConnCipher.Field(1).Elem().Elem()
			rf2InHalfConnCipherKs := rf2InHalfConnCipher.Field(0)

			rf2InHalfConnCipherKs = reflect.NewAt(rf2InHalfConnCipherKs.Type(), unsafe.Pointer(rf2InHalfConnCipherKs.UnsafeAddr())).Elem()
			rt2InHalfConnCipherKs := reflect.ValueOf(&c.InHalfConn.AEADAesGCM.Ks).Elem()
			rtw2InHalfConnCipherKs := reflect.NewAt(rt2InHalfConnCipherKs.Type(), unsafe.Pointer(rt2InHalfConnCipherKs.UnsafeAddr())).Elem()
			rtw2InHalfConnCipherKs.Set(rf2InHalfConnCipherKs)

			rf2InHalfConnCipherProductTable := rf2InHalfConnCipher.Field(1)
			rf2InHalfConnCipherProductTable = reflect.NewAt(rf2InHalfConnCipherProductTable.Type(), unsafe.Pointer(rf2InHalfConnCipherProductTable.UnsafeAddr())).Elem()
			rt2InHalfConnCipherProductTable := reflect.ValueOf(&c.InHalfConn.AEADAesGCM.ProductTable).Elem()
			rtw2InHalfConnCipherProductTable := reflect.NewAt(rt2InHalfConnCipherProductTable.Type(), unsafe.Pointer(rt2InHalfConnCipherProductTable.UnsafeAddr())).Elem()
			rtw2InHalfConnCipherProductTable.Set(rf2InHalfConnCipherProductTable)

			rf2InHalfConnCipherNonceSize := rf2InHalfConnCipher.Field(2)
			rf2InHalfConnCipherNonceSize = reflect.NewAt(rf2InHalfConnCipherNonceSize.Type(), unsafe.Pointer(rf2InHalfConnCipherNonceSize.UnsafeAddr())).Elem()
			rt2InHalfConnCipherNonceSize := reflect.ValueOf(&c.InHalfConn.AEADAesGCM.NonceSize).Elem()
			rtw2InHalfConnCipherNonceSize := reflect.NewAt(rt2InHalfConnCipherNonceSize.Type(), unsafe.Pointer(rt2InHalfConnCipherNonceSize.UnsafeAddr())).Elem()
			rtw2InHalfConnCipherNonceSize.Set(rf2InHalfConnCipherNonceSize)

			rf2InHalfConnCipherTagSize := rf2InHalfConnCipher.Field(3)
			rf2InHalfConnCipherTagSize = reflect.NewAt(rf2InHalfConnCipherTagSize.Type(), unsafe.Pointer(rf2InHalfConnCipherTagSize.UnsafeAddr())).Elem()
			rt2InHalfConnCipherTagSize := reflect.ValueOf(&c.InHalfConn.AEADAesGCM.TagSize).Elem()
			rtw2InHalfConnCipherTagSize := reflect.NewAt(rt2InHalfConnCipherTagSize.Type(), unsafe.Pointer(rt2InHalfConnCipherTagSize.UnsafeAddr())).Elem()
			rtw2InHalfConnCipherTagSize.Set(rf2InHalfConnCipherTagSize)
		}

	case "chacha20poly1305":
		{
			rf2InHalfConnCipher := rsInHalfConnCipher.Field(1).Elem().Elem()
			rf2InHalfConnCipherKey := rf2InHalfConnCipher.Field(0)

			rf2InHalfConnCipherKey = reflect.NewAt(rf2InHalfConnCipherKey.Type(), unsafe.Pointer(rf2InHalfConnCipherKey.UnsafeAddr())).Elem()
			rt2InHalfConnCipherKey := reflect.ValueOf(&c.InHalfConn.AEADChaCha.Key).Elem()
			rtw2InHalfConnCipherKey := reflect.NewAt(rt2InHalfConnCipherKey.Type(), unsafe.Pointer(rt2InHalfConnCipherKey.UnsafeAddr())).Elem()
			rtw2InHalfConnCipherKey.Set(rf2InHalfConnCipherKey)
		}
	default:
		// Unhandled
	}

	// Serialize the Out halfconn
	rsOutHalfConnCipher := reflect.ValueOf(c.Out.Cipher).Elem()
	rf1OutHalfConnCipher := rsOutHalfConnCipher.Field(0)
	rf1OutHalfConnCipher = reflect.NewAt(rf1OutHalfConnCipher.Type(), unsafe.Pointer(rf1OutHalfConnCipher.UnsafeAddr())).Elem()
	rt1OutHalfConnCipher := reflect.ValueOf(&c.OutHalfConn.NonceMask).Elem()
	rtw1OutHalfConnCipher := reflect.NewAt(rt1OutHalfConnCipher.Type(), unsafe.Pointer(rt1OutHalfConnCipher.UnsafeAddr())).Elem()
	rtw1OutHalfConnCipher.Set(rf1OutHalfConnCipher)

	switch rsOutHalfConnCipher.Field(1).Elem().Elem().Type().Name() {
	case "gcmAsm":
		{
			rf2OutHalfConnCipher := rsOutHalfConnCipher.Field(1).Elem().Elem()
			rf2OutHalfConnCipherKs := rf2OutHalfConnCipher.Field(0)

			rf2OutHalfConnCipherKs = reflect.NewAt(rf2OutHalfConnCipherKs.Type(), unsafe.Pointer(rf2OutHalfConnCipherKs.UnsafeAddr())).Elem()
			rt2OutHalfConnCipherKs := reflect.ValueOf(&c.OutHalfConn.AEADAesGCM.Ks).Elem()
			rtw2OutHalfConnCipherKs := reflect.NewAt(rt2OutHalfConnCipherKs.Type(), unsafe.Pointer(rt2OutHalfConnCipherKs.UnsafeAddr())).Elem()
			rtw2OutHalfConnCipherKs.Set(rf2OutHalfConnCipherKs)

			rf2OutHalfConnCipherProductTable := rf2OutHalfConnCipher.Field(1)
			rf2OutHalfConnCipherProductTable = reflect.NewAt(rf2OutHalfConnCipherProductTable.Type(), unsafe.Pointer(rf2OutHalfConnCipherProductTable.UnsafeAddr())).Elem()
			rt2OutHalfConnCipherProductTable := reflect.ValueOf(&c.OutHalfConn.AEADAesGCM.ProductTable).Elem()
			rtw2OutHalfConnCipherProductTable := reflect.NewAt(rt2OutHalfConnCipherProductTable.Type(), unsafe.Pointer(rt2OutHalfConnCipherProductTable.UnsafeAddr())).Elem()
			rtw2OutHalfConnCipherProductTable.Set(rf2OutHalfConnCipherProductTable)

			rf2OutHalfConnCipherNonceSize := rf2OutHalfConnCipher.Field(2)
			rf2OutHalfConnCipherNonceSize = reflect.NewAt(rf2OutHalfConnCipherNonceSize.Type(), unsafe.Pointer(rf2OutHalfConnCipherNonceSize.UnsafeAddr())).Elem()
			rt2OutHalfConnCipherNonceSize := reflect.ValueOf(&c.OutHalfConn.AEADAesGCM.NonceSize).Elem()
			rtw2OutHalfConnCipherNonceSize := reflect.NewAt(rt2OutHalfConnCipherNonceSize.Type(), unsafe.Pointer(rt2OutHalfConnCipherNonceSize.UnsafeAddr())).Elem()
			rtw2OutHalfConnCipherNonceSize.Set(rf2OutHalfConnCipherNonceSize)

			rf2OutHalfConnCipherTagSize := rf2OutHalfConnCipher.Field(3)
			rf2OutHalfConnCipherTagSize = reflect.NewAt(rf2OutHalfConnCipherTagSize.Type(), unsafe.Pointer(rf2OutHalfConnCipherTagSize.UnsafeAddr())).Elem()
			rt2OutHalfConnCipherTagSize := reflect.ValueOf(&c.OutHalfConn.AEADAesGCM.TagSize).Elem()
			rtw2OutHalfConnCipherTagSize := reflect.NewAt(rt2OutHalfConnCipherTagSize.Type(), unsafe.Pointer(rt2OutHalfConnCipherTagSize.UnsafeAddr())).Elem()
			rtw2OutHalfConnCipherTagSize.Set(rf2OutHalfConnCipherTagSize)
		}
	case "chacha20poly1305":
		{
			rf2OutHalfConnCipher := rsOutHalfConnCipher.Field(1).Elem().Elem()
			rf2OutHalfConnCipherKey := rf2OutHalfConnCipher.Field(0)

			rf2OutHalfConnCipherKey = reflect.NewAt(rf2OutHalfConnCipherKey.Type(), unsafe.Pointer(rf2OutHalfConnCipherKey.UnsafeAddr())).Elem()
			rt2OutHalfConnCipherKey := reflect.ValueOf(&c.OutHalfConn.AEADChaCha.Key).Elem()
			rtw2OutHalfConnCipherKey := reflect.NewAt(rt2OutHalfConnCipherKey.Type(), unsafe.Pointer(rt2OutHalfConnCipherKey.UnsafeAddr())).Elem()
			rtw2OutHalfConnCipherKey.Set(rf2OutHalfConnCipherKey)
		}
	}

	hs.transcript.Write(finishedMsg.marshal())

	if !hs.shouldSendSessionTickets() {
		return nil
	}

	resumptionSecret := hs.suite.deriveSecret(hs.masterSecret,
		resumptionLabel, hs.transcript)

	m := new(newSessionTicketMsgTLS13)

	var certsFromClient [][]byte
	for _, cert := range c.peerCertificates {
		certsFromClient = append(certsFromClient, cert.Raw)
	}
	state := sessionStateTLS13{
		cipherSuite:      hs.suite.Id,
		createdAt:        uint64(c.config.time().Unix()),
		resumptionSecret: resumptionSecret,
		certificate: Certificate{
			Certificate:                 certsFromClient,
			OCSPStaple:                  c.OcspResponse,
			SignedCertificateTimestamps: c.Scts,
		},
	}
	var err error
	m.label, err = c.encryptTicket(state.marshal())
	if err != nil {
		return err
	}
	m.lifetime = uint32(maxSessionTicketLifetime / time.Second)

	if _, err := c.writeRecord(recordTypeHandshake, m.marshal()); err != nil {
		return err
	}

	return nil
}

func (hs *serverHandshakeStateTLS13) readClientCertificate() error {
	c := hs.c

	if !hs.requestClientCert() {
		// Make sure the connection is still being verified whether or not
		// the server requested a client certificate.
		if c.config.VerifyConnection != nil {
			if err := c.config.VerifyConnection(c.connectionStateLocked()); err != nil {
				c.sendAlert(alertBadCertificate)
				return err
			}
		}
		return nil
	}

	// If we requested a client certificate, then the client must send a
	// certificate message. If it's empty, no CertificateVerify is sent.

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	certMsg, ok := msg.(*certificateMsgTLS13)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(certMsg, msg)
	}
	hs.transcript.Write(certMsg.marshal())

	if err := c.processCertsFromClient(certMsg.certificate); err != nil {
		return err
	}

	if c.config.VerifyConnection != nil {
		if err := c.config.VerifyConnection(c.connectionStateLocked()); err != nil {
			c.sendAlert(alertBadCertificate)
			return err
		}
	}

	if len(certMsg.certificate.Certificate) != 0 {
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
		if !isSupportedSignatureAlgorithm(certVerify.signatureAlgorithm, supportedSignatureAlgorithms) {
			c.sendAlert(alertIllegalParameter)
			return errors.New("tls: client certificate used with invalid signature algorithm")
		}
		sigType, sigHash, err := typeAndHashFromSignatureScheme(certVerify.signatureAlgorithm)
		if err != nil {
			return c.sendAlert(alertInternalError)
		}
		if sigType == signaturePKCS1v15 || sigHash == crypto.SHA1 {
			c.sendAlert(alertIllegalParameter)
			return errors.New("tls: client certificate used with invalid signature algorithm")
		}
		signed := signedMessage(sigHash, clientSignatureContext, hs.transcript)
		if err := verifyHandshakeSignature(sigType, c.peerCertificates[0].PublicKey,
			sigHash, signed, certVerify.signature); err != nil {
			c.sendAlert(alertDecryptError)
			return errors.New("tls: invalid signature by the client certificate: " + err.Error())
		}

		hs.transcript.Write(certVerify.marshal())
	}

	// If we waited until the client certificates to send session tickets, we
	// are ready to do it now.
	if err := hs.sendSessionTickets(); err != nil {
		return err
	}

	return nil
}

func (hs *serverHandshakeStateTLS13) readClientFinished() error {
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

	if !hmac.Equal(hs.clientFinished, finished.verifyData) {
		c.sendAlert(alertDecryptError)
		return errors.New("tls: invalid client finished hash")
	}

	c.In.setTrafficSecret(hs.suite, hs.trafficSecret)

	return nil
}
