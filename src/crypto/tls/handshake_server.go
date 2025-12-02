// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rsa"
	"crypto/subtle"
	"crypto/tls/internal/fips140tls"
	"crypto/x509"
	"errors"
	"fmt"
	"hash"
	"io"
	"time"
)

// serverHandshakeState contains details of a server handshake in progress.
// It's discarded once the handshake has completed.
type serverHandshakeState struct {
	c            *Conn
	ctx          context.Context
	clientHello  *clientHelloMsg
	hello        *serverHelloMsg
	suite        *cipherSuite
	ecdheOk      bool
	ecSignOk     bool
	rsaDecryptOk bool
	rsaSignOk    bool
	sessionState *SessionState
	finishedHash finishedHash
	masterSecret []byte
	cert         *Certificate
}

// serverHandshake performs a TLS handshake as a server.
func (c *Conn) serverHandshake(ctx context.Context) error {
	clientHello, ech, err := c.readClientHello(ctx)
	if err != nil {
		return err
	}

	if c.vers == VersionTLS13 {
		hs := serverHandshakeStateTLS13{
			c:           c,
			ctx:         ctx,
			clientHello: clientHello,
			echContext:  ech,
		}
		return hs.handshake()
	}

	hs := serverHandshakeState{
		c:           c,
		ctx:         ctx,
		clientHello: clientHello,
	}
	return hs.handshake()
}

func (hs *serverHandshakeState) handshake() error {
	c := hs.c

	if err := hs.processClientHello(); err != nil {
		return err
	}

	// For an overview of TLS handshaking, see RFC 5246, Section 7.3.
	c.buffering = true
	if err := hs.checkForResumption(); err != nil {
		return err
	}
	if hs.sessionState != nil {
		// The client has included a session ticket and so we do an abbreviated handshake.
		if err := hs.doResumeHandshake(); err != nil {
			return err
		}
		if err := hs.establishKeys(); err != nil {
			return err
		}
		if err := hs.sendSessionTicket(); err != nil {
			return err
		}
		if err := hs.sendFinished(c.serverFinished[:]); err != nil {
			return err
		}
		if _, err := c.flush(); err != nil {
			return err
		}
		c.clientFinishedIsFirst = false
		if err := hs.readFinished(nil); err != nil {
			return err
		}
	} else {
		// The client didn't include a session ticket, or it wasn't
		// valid so we do a full handshake.
		if err := hs.pickCipherSuite(); err != nil {
			return err
		}
		if err := hs.doFullHandshake(); err != nil {
			return err
		}
		if err := hs.establishKeys(); err != nil {
			return err
		}
		if err := hs.readFinished(c.clientFinished[:]); err != nil {
			return err
		}
		c.clientFinishedIsFirst = true
		c.buffering = true
		if err := hs.sendSessionTicket(); err != nil {
			return err
		}
		if err := hs.sendFinished(nil); err != nil {
			return err
		}
		if _, err := c.flush(); err != nil {
			return err
		}
	}

	c.ekm = ekmFromMasterSecret(c.vers, hs.suite, hs.masterSecret, hs.clientHello.random, hs.hello.random)
	c.isHandshakeComplete.Store(true)

	return nil
}

// readClientHello reads a ClientHello message and selects the protocol version.
func (c *Conn) readClientHello(ctx context.Context) (*clientHelloMsg, *echServerContext, error) {
	// clientHelloMsg is included in the transcript, but we haven't initialized
	// it yet. The respective handshake functions will record it themselves.
	msg, err := c.readHandshake(nil)
	if err != nil {
		return nil, nil, err
	}
	clientHello, ok := msg.(*clientHelloMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return nil, nil, unexpectedMessageError(clientHello, msg)
	}

	// ECH processing has to be done before we do any other negotiation based on
	// the contents of the client hello, since we may swap it out completely.
	var ech *echServerContext
	if len(clientHello.encryptedClientHello) != 0 {
		echKeys := c.config.EncryptedClientHelloKeys
		if c.config.GetEncryptedClientHelloKeys != nil {
			echKeys, err = c.config.GetEncryptedClientHelloKeys(clientHelloInfo(ctx, c, clientHello))
			if err != nil {
				c.sendAlert(alertInternalError)
				return nil, nil, err
			}
		}
		clientHello, ech, err = c.processECHClientHello(clientHello, echKeys)
		if err != nil {
			return nil, nil, err
		}
	}

	var configForClient *Config
	originalConfig := c.config
	if c.config.GetConfigForClient != nil {
		chi := clientHelloInfo(ctx, c, clientHello)
		if configForClient, err = c.config.GetConfigForClient(chi); err != nil {
			c.sendAlert(alertInternalError)
			return nil, nil, err
		} else if configForClient != nil {
			c.config = configForClient
		}
	}
	c.ticketKeys = originalConfig.ticketKeys(configForClient)

	clientVersions := clientHello.supportedVersions
	if clientHello.vers >= VersionTLS13 && len(clientVersions) == 0 {
		// RFC 8446 4.2.1 indicates when the supported_versions extension is not sent,
		// compatible servers MUST negotiate TLS 1.2 or earlier if supported, even
		// if the client legacy version is TLS 1.3 or later.
		//
		// Since we reject empty extensionSupportedVersions in the client hello unmarshal
		// finding the supportedVersions empty indicates the extension was not present.
		clientVersions = supportedVersionsFromMax(VersionTLS12)
	} else if len(clientVersions) == 0 {
		clientVersions = supportedVersionsFromMax(clientHello.vers)
	}
	c.vers, ok = c.config.mutualVersion(roleServer, clientVersions)
	if !ok {
		c.sendAlert(alertProtocolVersion)
		return nil, nil, fmt.Errorf("tls: client offered only unsupported versions: %x", clientVersions)
	}
	c.haveVers = true
	c.in.version = c.vers
	c.out.version = c.vers

	// This check reflects some odd specification implied behavior. Client-facing servers
	// are supposed to reject hellos with outer ECH and inner ECH that offers 1.2, but
	// backend servers are allowed to accept hellos with inner ECH that offer 1.2, since
	// they cannot expect client-facing servers to behave properly. Since we act as both
	// a client-facing and backend server, we only enforce 1.3 being negotiated if we
	// saw a hello with outer ECH first. The spec probably should've made this an error,
	// but it didn't, and this matches the boringssl behavior.
	if c.vers != VersionTLS13 && (ech != nil && !ech.inner) {
		c.sendAlert(alertIllegalParameter)
		return nil, nil, errors.New("tls: Encrypted Client Hello cannot be used pre-TLS 1.3")
	}

	if c.config.MinVersion == 0 && c.vers < VersionTLS12 {
		tls10server.Value() // ensure godebug is initialized
		tls10server.IncNonDefault()
	}

	return clientHello, ech, nil
}

func (hs *serverHandshakeState) processClientHello() error {
	c := hs.c

	hs.hello = new(serverHelloMsg)
	hs.hello.vers = c.vers

	foundCompression := false
	// We only support null compression, so check that the client offered it.
	for _, compression := range hs.clientHello.compressionMethods {
		if compression == compressionNone {
			foundCompression = true
			break
		}
	}

	if !foundCompression {
		c.sendAlert(alertIllegalParameter)
		return errors.New("tls: client does not support uncompressed connections")
	}

	hs.hello.random = make([]byte, 32)
	serverRandom := hs.hello.random
	// Downgrade protection canaries. See RFC 8446, Section 4.1.3.
	maxVers := c.config.maxSupportedVersion(roleServer)
	if maxVers >= VersionTLS12 && c.vers < maxVers || testingOnlyForceDowngradeCanary {
		if c.vers == VersionTLS12 {
			copy(serverRandom[24:], downgradeCanaryTLS12)
		} else {
			copy(serverRandom[24:], downgradeCanaryTLS11)
		}
		serverRandom = serverRandom[:24]
	}
	_, err := io.ReadFull(c.config.rand(), serverRandom)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}

	if len(hs.clientHello.secureRenegotiation) != 0 {
		c.sendAlert(alertHandshakeFailure)
		return errors.New("tls: initial handshake had non-empty renegotiation extension")
	}

	hs.hello.extendedMasterSecret = hs.clientHello.extendedMasterSecret
	hs.hello.secureRenegotiationSupported = hs.clientHello.secureRenegotiationSupported
	hs.hello.compressionMethod = compressionNone
	if len(hs.clientHello.serverName) > 0 {
		c.serverName = hs.clientHello.serverName
	}

	selectedProto, err := negotiateALPN(c.config.NextProtos, hs.clientHello.alpnProtocols, false)
	if err != nil {
		c.sendAlert(alertNoApplicationProtocol)
		return err
	}
	hs.hello.alpnProtocol = selectedProto
	c.clientProtocol = selectedProto

	hs.cert, err = c.config.getCertificate(clientHelloInfo(hs.ctx, c, hs.clientHello))
	if err != nil {
		if err == errNoCertificates {
			c.sendAlert(alertUnrecognizedName)
		} else {
			c.sendAlert(alertInternalError)
		}
		return err
	}
	if hs.clientHello.scts {
		hs.hello.scts = hs.cert.SignedCertificateTimestamps
	}

	hs.ecdheOk, err = supportsECDHE(c.config, c.vers, hs.clientHello.supportedCurves, hs.clientHello.supportedPoints)
	if err != nil {
		c.sendAlert(alertMissingExtension)
		return err
	}

	if hs.ecdheOk && len(hs.clientHello.supportedPoints) > 0 {
		// Although omitting the ec_point_formats extension is permitted, some
		// old OpenSSL version will refuse to handshake if not present.
		//
		// Per RFC 4492, section 5.1.2, implementations MUST support the
		// uncompressed point format. See golang.org/issue/31943.
		hs.hello.supportedPoints = []uint8{pointFormatUncompressed}
	}

	if priv, ok := hs.cert.PrivateKey.(crypto.Signer); ok {
		switch priv.Public().(type) {
		case *ecdsa.PublicKey:
			hs.ecSignOk = true
		case ed25519.PublicKey:
			hs.ecSignOk = true
		case *rsa.PublicKey:
			hs.rsaSignOk = true
		default:
			c.sendAlert(alertInternalError)
			return fmt.Errorf("tls: unsupported signing key type (%T)", priv.Public())
		}
	}
	if priv, ok := hs.cert.PrivateKey.(crypto.Decrypter); ok {
		switch priv.Public().(type) {
		case *rsa.PublicKey:
			hs.rsaDecryptOk = true
		default:
			c.sendAlert(alertInternalError)
			return fmt.Errorf("tls: unsupported decryption key type (%T)", priv.Public())
		}
	}

	return nil
}

// negotiateALPN picks a shared ALPN protocol that both sides support in server
// preference order. If ALPN is not configured or the peer doesn't support it,
// it returns "" and no error.
func negotiateALPN(serverProtos, clientProtos []string, quic bool) (string, error) {
	if len(serverProtos) == 0 || len(clientProtos) == 0 {
		if quic && len(serverProtos) != 0 {
			// RFC 9001, Section 8.1
			return "", fmt.Errorf("tls: client did not request an application protocol")
		}
		return "", nil
	}
	var http11fallback bool
	for _, s := range serverProtos {
		for _, c := range clientProtos {
			if s == c {
				return s, nil
			}
			if s == "h2" && c == "http/1.1" {
				http11fallback = true
			}
		}
	}
	// As a special case, let http/1.1 clients connect to h2 servers as if they
	// didn't support ALPN. We used not to enforce protocol overlap, so over
	// time a number of HTTP servers were configured with only "h2", but
	// expected to accept connections from "http/1.1" clients. See Issue 46310.
	if http11fallback {
		return "", nil
	}
	return "", fmt.Errorf("tls: client requested unsupported application protocols (%q)", clientProtos)
}

// supportsECDHE returns whether ECDHE key exchanges can be used with this
// pre-TLS 1.3 client.
func supportsECDHE(c *Config, version uint16, supportedCurves []CurveID, supportedPoints []uint8) (bool, error) {
	supportsCurve := false
	for _, curve := range supportedCurves {
		if c.supportsCurve(version, curve) {
			supportsCurve = true
			break
		}
	}

	supportsPointFormat := false
	offeredNonCompressedFormat := false
	for _, pointFormat := range supportedPoints {
		if pointFormat == pointFormatUncompressed {
			supportsPointFormat = true
		} else {
			offeredNonCompressedFormat = true
		}
	}
	// Per RFC 8422, Section 5.1.2, if the Supported Point Formats extension is
	// missing, uncompressed points are supported. If supportedPoints is empty,
	// the extension must be missing, as an empty extension body is rejected by
	// the parser. See https://go.dev/issue/49126.
	if len(supportedPoints) == 0 {
		supportsPointFormat = true
	} else if offeredNonCompressedFormat && !supportsPointFormat {
		return false, errors.New("tls: client offered only incompatible point formats")
	}

	return supportsCurve && supportsPointFormat, nil
}

func (hs *serverHandshakeState) pickCipherSuite() error {
	c := hs.c

	preferenceList := c.config.cipherSuites(isAESGCMPreferred(hs.clientHello.cipherSuites))

	hs.suite = selectCipherSuite(preferenceList, hs.clientHello.cipherSuites, hs.cipherSuiteOk)
	if hs.suite == nil {
		c.sendAlert(alertHandshakeFailure)
		return fmt.Errorf("tls: no cipher suite supported by both client and server; client offered: %x",
			hs.clientHello.cipherSuites)
	}
	c.cipherSuite = hs.suite.id

	if c.config.CipherSuites == nil && !fips140tls.Required() && rsaKexCiphers[hs.suite.id] {
		tlsrsakex.Value() // ensure godebug is initialized
		tlsrsakex.IncNonDefault()
	}
	if c.config.CipherSuites == nil && !fips140tls.Required() && tdesCiphers[hs.suite.id] {
		tls3des.Value() // ensure godebug is initialized
		tls3des.IncNonDefault()
	}

	for _, id := range hs.clientHello.cipherSuites {
		if id == TLS_FALLBACK_SCSV {
			// The client is doing a fallback connection. See RFC 7507.
			if hs.clientHello.vers < c.config.maxSupportedVersion(roleServer) {
				c.sendAlert(alertInappropriateFallback)
				return errors.New("tls: client using inappropriate protocol fallback")
			}
			break
		}
	}

	return nil
}

func (hs *serverHandshakeState) cipherSuiteOk(c *cipherSuite) bool {
	if c.flags&suiteECDHE != 0 {
		if !hs.ecdheOk {
			return false
		}
		if c.flags&suiteECSign != 0 {
			if !hs.ecSignOk {
				return false
			}
		} else if !hs.rsaSignOk {
			return false
		}
	} else if !hs.rsaDecryptOk {
		return false
	}
	if hs.c.vers < VersionTLS12 && c.flags&suiteTLS12 != 0 {
		return false
	}
	return true
}

// checkForResumption reports whether we should perform resumption on this connection.
func (hs *serverHandshakeState) checkForResumption() error {
	c := hs.c

	if c.config.SessionTicketsDisabled {
		return nil
	}

	var sessionState *SessionState
	if c.config.UnwrapSession != nil {
		ss, err := c.config.UnwrapSession(hs.clientHello.sessionTicket, c.connectionStateLocked())
		if err != nil {
			return err
		}
		if ss == nil {
			return nil
		}
		sessionState = ss
	} else {
		plaintext := c.config.decryptTicket(hs.clientHello.sessionTicket, c.ticketKeys)
		if plaintext == nil {
			return nil
		}
		ss, err := ParseSessionState(plaintext)
		if err != nil {
			return nil
		}
		sessionState = ss
	}

	// TLS 1.2 tickets don't natively have a lifetime, but we want to avoid
	// re-wrapping the same master secret in different tickets over and over for
	// too long, weakening forward secrecy.
	createdAt := time.Unix(int64(sessionState.createdAt), 0)
	if c.config.time().Sub(createdAt) > maxSessionTicketLifetime {
		return nil
	}

	// Never resume a session for a different TLS version.
	if c.vers != sessionState.version {
		return nil
	}

	cipherSuiteOk := false
	// Check that the client is still offering the ciphersuite in the session.
	for _, id := range hs.clientHello.cipherSuites {
		if id == sessionState.cipherSuite {
			cipherSuiteOk = true
			break
		}
	}
	if !cipherSuiteOk {
		return nil
	}

	// Check that we also support the ciphersuite from the session.
	suite := selectCipherSuite([]uint16{sessionState.cipherSuite},
		c.config.supportedCipherSuites(), hs.cipherSuiteOk)
	if suite == nil {
		return nil
	}

	sessionHasClientCerts := len(sessionState.peerCertificates) != 0
	needClientCerts := requiresClientCert(c.config.ClientAuth)
	if needClientCerts && !sessionHasClientCerts {
		return nil
	}
	if sessionHasClientCerts && c.config.ClientAuth == NoClientCert {
		return nil
	}
	if sessionHasClientCerts && c.config.time().After(sessionState.peerCertificates[0].NotAfter) {
		return nil
	}
	if sessionHasClientCerts && c.config.ClientAuth >= VerifyClientCertIfGiven &&
		len(sessionState.verifiedChains) == 0 {
		return nil
	}

	// RFC 7627, Section 5.3
	if !sessionState.extMasterSecret && hs.clientHello.extendedMasterSecret {
		return nil
	}
	if sessionState.extMasterSecret && !hs.clientHello.extendedMasterSecret {
		// Aborting is somewhat harsh, but it's a MUST and it would indicate a
		// weird downgrade in client capabilities.
		return errors.New("tls: session supported extended_master_secret but client does not")
	}
	if !sessionState.extMasterSecret && fips140tls.Required() {
		// FIPS 140-3 requires the use of Extended Master Secret.
		return nil
	}

	c.peerCertificates = sessionState.peerCertificates
	c.ocspResponse = sessionState.ocspResponse
	c.scts = sessionState.scts
	c.verifiedChains = sessionState.verifiedChains
	c.extMasterSecret = sessionState.extMasterSecret
	hs.sessionState = sessionState
	hs.suite = suite
	c.curveID = sessionState.curveID
	c.didResume = true
	return nil
}

func (hs *serverHandshakeState) doResumeHandshake() error {
	c := hs.c

	hs.hello.cipherSuite = hs.suite.id
	c.cipherSuite = hs.suite.id
	// We echo the client's session ID in the ServerHello to let it know
	// that we're doing a resumption.
	hs.hello.sessionId = hs.clientHello.sessionId
	// We always send a new session ticket, even if it wraps the same master
	// secret and it's potentially encrypted with the same key, to help the
	// client avoid cross-connection tracking from a network observer.
	hs.hello.ticketSupported = true
	hs.finishedHash = newFinishedHash(c.vers, hs.suite)
	hs.finishedHash.discardHandshakeBuffer()
	if err := transcriptMsg(hs.clientHello, &hs.finishedHash); err != nil {
		return err
	}
	if _, err := hs.c.writeHandshakeRecord(hs.hello, &hs.finishedHash); err != nil {
		return err
	}

	if c.config.VerifyConnection != nil {
		if err := c.config.VerifyConnection(c.connectionStateLocked()); err != nil {
			c.sendAlert(alertBadCertificate)
			return err
		}
	}

	hs.masterSecret = hs.sessionState.secret

	return nil
}

func (hs *serverHandshakeState) doFullHandshake() error {
	c := hs.c

	if hs.clientHello.ocspStapling && len(hs.cert.OCSPStaple) > 0 {
		hs.hello.ocspStapling = true
	}

	if hs.clientHello.serverName != "" {
		hs.hello.serverNameAck = true
	}

	hs.hello.ticketSupported = hs.clientHello.ticketSupported && !c.config.SessionTicketsDisabled
	hs.hello.cipherSuite = hs.suite.id

	hs.finishedHash = newFinishedHash(hs.c.vers, hs.suite)
	if c.config.ClientAuth == NoClientCert {
		// No need to keep a full record of the handshake if client
		// certificates won't be used.
		hs.finishedHash.discardHandshakeBuffer()
	}
	if err := transcriptMsg(hs.clientHello, &hs.finishedHash); err != nil {
		return err
	}
	if _, err := hs.c.writeHandshakeRecord(hs.hello, &hs.finishedHash); err != nil {
		return err
	}

	certMsg := new(certificateMsg)
	certMsg.certificates = hs.cert.Certificate
	if _, err := hs.c.writeHandshakeRecord(certMsg, &hs.finishedHash); err != nil {
		return err
	}

	if hs.hello.ocspStapling {
		certStatus := new(certificateStatusMsg)
		certStatus.response = hs.cert.OCSPStaple
		if _, err := hs.c.writeHandshakeRecord(certStatus, &hs.finishedHash); err != nil {
			return err
		}
	}

	keyAgreement := hs.suite.ka(c.vers)
	skx, err := keyAgreement.generateServerKeyExchange(c.config, hs.cert, hs.clientHello, hs.hello)
	if err != nil {
		c.sendAlert(alertHandshakeFailure)
		return err
	}
	if skx != nil {
		if keyAgreement, ok := keyAgreement.(*ecdheKeyAgreement); ok {
			c.curveID = keyAgreement.curveID
			c.peerSigAlg = keyAgreement.signatureAlgorithm
		}
		if _, err := hs.c.writeHandshakeRecord(skx, &hs.finishedHash); err != nil {
			return err
		}
	}

	var certReq *certificateRequestMsg
	if c.config.ClientAuth >= RequestClientCert {
		// Request a client certificate
		certReq = new(certificateRequestMsg)
		certReq.certificateTypes = []byte{
			byte(certTypeRSASign),
			byte(certTypeECDSASign),
		}
		if c.vers >= VersionTLS12 {
			certReq.hasSignatureAlgorithm = true
			certReq.supportedSignatureAlgorithms = supportedSignatureAlgorithms(c.vers)
		}

		// An empty list of certificateAuthorities signals to
		// the client that it may send any certificate in response
		// to our request. When we know the CAs we trust, then
		// we can send them down, so that the client can choose
		// an appropriate certificate to give to us.
		if c.config.ClientCAs != nil {
			certReq.certificateAuthorities = c.config.ClientCAs.Subjects()
		}
		if _, err := hs.c.writeHandshakeRecord(certReq, &hs.finishedHash); err != nil {
			return err
		}
	}

	helloDone := new(serverHelloDoneMsg)
	if _, err := hs.c.writeHandshakeRecord(helloDone, &hs.finishedHash); err != nil {
		return err
	}

	if _, err := c.flush(); err != nil {
		return err
	}

	var pub crypto.PublicKey // public key for client auth, if any

	msg, err := c.readHandshake(&hs.finishedHash)
	if err != nil {
		return err
	}

	// If we requested a client certificate, then the client must send a
	// certificate message, even if it's empty.
	if c.config.ClientAuth >= RequestClientCert {
		certMsg, ok := msg.(*certificateMsg)
		if !ok {
			c.sendAlert(alertUnexpectedMessage)
			return unexpectedMessageError(certMsg, msg)
		}

		if err := c.processCertsFromClient(Certificate{
			Certificate: certMsg.certificates,
		}); err != nil {
			return err
		}
		if len(certMsg.certificates) != 0 {
			pub = c.peerCertificates[0].PublicKey
		}

		msg, err = c.readHandshake(&hs.finishedHash)
		if err != nil {
			return err
		}
	}
	if c.config.VerifyConnection != nil {
		if err := c.config.VerifyConnection(c.connectionStateLocked()); err != nil {
			c.sendAlert(alertBadCertificate)
			return err
		}
	}

	// Get client key exchange
	ckx, ok := msg.(*clientKeyExchangeMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(ckx, msg)
	}

	preMasterSecret, err := keyAgreement.processClientKeyExchange(c.config, hs.cert, ckx, c.vers)
	if err != nil {
		c.sendAlert(alertIllegalParameter)
		return err
	}
	if hs.hello.extendedMasterSecret {
		c.extMasterSecret = true
		hs.masterSecret = extMasterFromPreMasterSecret(c.vers, hs.suite, preMasterSecret,
			hs.finishedHash.Sum())
	} else {
		if fips140tls.Required() {
			c.sendAlert(alertHandshakeFailure)
			return errors.New("tls: FIPS 140-3 requires the use of Extended Master Secret")
		}
		hs.masterSecret = masterFromPreMasterSecret(c.vers, hs.suite, preMasterSecret,
			hs.clientHello.random, hs.hello.random)
	}
	if err := c.config.writeKeyLog(keyLogLabelTLS12, hs.clientHello.random, hs.masterSecret); err != nil {
		c.sendAlert(alertInternalError)
		return err
	}

	// If we received a client cert in response to our certificate request message,
	// the client will send us a certificateVerifyMsg immediately after the
	// clientKeyExchangeMsg. This message is a digest of all preceding
	// handshake-layer messages that is signed using the private key corresponding
	// to the client's certificate. This allows us to verify that the client is in
	// possession of the private key of the certificate.
	if len(c.peerCertificates) > 0 {
		// certificateVerifyMsg is included in the transcript, but not until
		// after we verify the handshake signature, since the state before
		// this message was sent is used.
		msg, err = c.readHandshake(nil)
		if err != nil {
			return err
		}
		certVerify, ok := msg.(*certificateVerifyMsg)
		if !ok {
			c.sendAlert(alertUnexpectedMessage)
			return unexpectedMessageError(certVerify, msg)
		}

		var sigType uint8
		var sigHash crypto.Hash
		if c.vers >= VersionTLS12 {
			if !isSupportedSignatureAlgorithm(certVerify.signatureAlgorithm, certReq.supportedSignatureAlgorithms) {
				c.sendAlert(alertIllegalParameter)
				return errors.New("tls: client certificate used with invalid signature algorithm")
			}
			sigType, sigHash, err = typeAndHashFromSignatureScheme(certVerify.signatureAlgorithm)
			if err != nil {
				return c.sendAlert(alertInternalError)
			}
			if sigHash == crypto.SHA1 {
				tlssha1.Value() // ensure godebug is initialized
				tlssha1.IncNonDefault()
			}
			if hs.finishedHash.buffer == nil {
				c.sendAlert(alertInternalError)
				return errors.New("tls: internal error: did not keep handshake transcript for TLS 1.2")
			}
			if err := verifyHandshakeSignature(sigType, pub, sigHash, hs.finishedHash.buffer, certVerify.signature); err != nil {
				c.sendAlert(alertDecryptError)
				return errors.New("tls: invalid signature by the client certificate: " + err.Error())
			}
		} else {
			sigType, sigHash, err = legacyTypeAndHashFromPublicKey(pub)
			if err != nil {
				c.sendAlert(alertIllegalParameter)
				return err
			}
			signed := hs.finishedHash.hashForClientCertificate(sigType)
			if err := verifyLegacyHandshakeSignature(sigType, pub, sigHash, signed, certVerify.signature); err != nil {
				c.sendAlert(alertDecryptError)
				return errors.New("tls: invalid signature by the client certificate: " + err.Error())
			}
		}

		c.peerSigAlg = certVerify.signatureAlgorithm

		if err := transcriptMsg(certVerify, &hs.finishedHash); err != nil {
			return err
		}
	}

	hs.finishedHash.discardHandshakeBuffer()

	return nil
}

func (hs *serverHandshakeState) establishKeys() error {
	c := hs.c

	clientMAC, serverMAC, clientKey, serverKey, clientIV, serverIV :=
		keysFromMasterSecret(c.vers, hs.suite, hs.masterSecret, hs.clientHello.random, hs.hello.random, hs.suite.macLen, hs.suite.keyLen, hs.suite.ivLen)

	var clientCipher, serverCipher any
	var clientHash, serverHash hash.Hash

	if hs.suite.aead == nil {
		clientCipher = hs.suite.cipher(clientKey, clientIV, true /* for reading */)
		clientHash = hs.suite.mac(clientMAC)
		serverCipher = hs.suite.cipher(serverKey, serverIV, false /* not for reading */)
		serverHash = hs.suite.mac(serverMAC)
	} else {
		clientCipher = hs.suite.aead(clientKey, clientIV)
		serverCipher = hs.suite.aead(serverKey, serverIV)
	}

	c.in.prepareCipherSpec(c.vers, clientCipher, clientHash)
	c.out.prepareCipherSpec(c.vers, serverCipher, serverHash)

	return nil
}

func (hs *serverHandshakeState) readFinished(out []byte) error {
	c := hs.c

	if err := c.readChangeCipherSpec(); err != nil {
		return err
	}

	// finishedMsg is included in the transcript, but not until after we
	// check the client version, since the state before this message was
	// sent is used during verification.
	msg, err := c.readHandshake(nil)
	if err != nil {
		return err
	}
	clientFinished, ok := msg.(*finishedMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(clientFinished, msg)
	}

	verify := hs.finishedHash.clientSum(hs.masterSecret)
	if len(verify) != len(clientFinished.verifyData) ||
		subtle.ConstantTimeCompare(verify, clientFinished.verifyData) != 1 {
		c.sendAlert(alertHandshakeFailure)
		return errors.New("tls: client's Finished message is incorrect")
	}

	if err := transcriptMsg(clientFinished, &hs.finishedHash); err != nil {
		return err
	}

	copy(out, verify)
	return nil
}

func (hs *serverHandshakeState) sendSessionTicket() error {
	if !hs.hello.ticketSupported {
		return nil
	}

	c := hs.c
	m := new(newSessionTicketMsg)

	state := c.sessionState()
	state.secret = hs.masterSecret
	if hs.sessionState != nil {
		// If this is re-wrapping an old key, then keep
		// the original time it was created.
		state.createdAt = hs.sessionState.createdAt
	}
	if c.config.WrapSession != nil {
		var err error
		m.ticket, err = c.config.WrapSession(c.connectionStateLocked(), state)
		if err != nil {
			return err
		}
	} else {
		stateBytes, err := state.Bytes()
		if err != nil {
			return err
		}
		m.ticket, err = c.config.encryptTicket(stateBytes, c.ticketKeys)
		if err != nil {
			return err
		}
	}

	if _, err := hs.c.writeHandshakeRecord(m, &hs.finishedHash); err != nil {
		return err
	}

	return nil
}

func (hs *serverHandshakeState) sendFinished(out []byte) error {
	c := hs.c

	if err := c.writeChangeCipherRecord(); err != nil {
		return err
	}

	finished := new(finishedMsg)
	finished.verifyData = hs.finishedHash.serverSum(hs.masterSecret)
	if _, err := hs.c.writeHandshakeRecord(finished, &hs.finishedHash); err != nil {
		return err
	}

	copy(out, finished.verifyData)

	return nil
}

// processCertsFromClient takes a chain of client certificates either from a
// certificateMsg message or a certificateMsgTLS13 message and verifies them.
func (c *Conn) processCertsFromClient(certificate Certificate) error {
	certificates := certificate.Certificate
	certs := make([]*x509.Certificate, len(certificates))
	var err error
	for i, asn1Data := range certificates {
		if certs[i], err = x509.ParseCertificate(asn1Data); err != nil {
			c.sendAlert(alertDecodeError)
			return errors.New("tls: failed to parse client certificate: " + err.Error())
		}
		if certs[i].PublicKeyAlgorithm == x509.RSA {
			n := certs[i].PublicKey.(*rsa.PublicKey).N.BitLen()
			if max, ok := checkKeySize(n); !ok {
				c.sendAlert(alertBadCertificate)
				return fmt.Errorf("tls: client sent certificate containing RSA key larger than %d bits", max)
			}
		}
	}

	if len(certs) == 0 && requiresClientCert(c.config.ClientAuth) {
		if c.vers == VersionTLS13 {
			c.sendAlert(alertCertificateRequired)
		} else {
			c.sendAlert(alertHandshakeFailure)
		}
		return errors.New("tls: client didn't provide a certificate")
	}

	if c.config.ClientAuth >= VerifyClientCertIfGiven && len(certs) > 0 {
		opts := x509.VerifyOptions{
			Roots:         c.config.ClientCAs,
			CurrentTime:   c.config.time(),
			Intermediates: x509.NewCertPool(),
			KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}

		for _, cert := range certs[1:] {
			opts.Intermediates.AddCert(cert)
		}

		chains, err := certs[0].Verify(opts)
		if err != nil {
			if _, ok := errors.AsType[x509.UnknownAuthorityError](err); ok {
				c.sendAlert(alertUnknownCA)
			} else if errCertificateInvalid, ok := errors.AsType[x509.CertificateInvalidError](err); ok && errCertificateInvalid.Reason == x509.Expired {
				c.sendAlert(alertCertificateExpired)
			} else {
				c.sendAlert(alertBadCertificate)
			}
			return &CertificateVerificationError{UnverifiedCertificates: certs, Err: err}
		}

		c.verifiedChains, err = fipsAllowedChains(chains)
		if err != nil {
			c.sendAlert(alertBadCertificate)
			return &CertificateVerificationError{UnverifiedCertificates: certs, Err: err}
		}
	}

	c.peerCertificates = certs
	c.ocspResponse = certificate.OCSPStaple
	c.scts = certificate.SignedCertificateTimestamps

	if len(certs) > 0 {
		switch certs[0].PublicKey.(type) {
		case *ecdsa.PublicKey, *rsa.PublicKey, ed25519.PublicKey:
		default:
			c.sendAlert(alertUnsupportedCertificate)
			return fmt.Errorf("tls: client certificate contains an unsupported public key of type %T", certs[0].PublicKey)
		}
	}

	if c.config.VerifyPeerCertificate != nil {
		if err := c.config.VerifyPeerCertificate(certificates, c.verifiedChains); err != nil {
			c.sendAlert(alertBadCertificate)
			return err
		}
	}

	return nil
}

func clientHelloInfo(ctx context.Context, c *Conn, clientHello *clientHelloMsg) *ClientHelloInfo {
	supportedVersions := clientHello.supportedVersions
	if len(clientHello.supportedVersions) == 0 {
		supportedVersions = supportedVersionsFromMax(clientHello.vers)
	}

	return &ClientHelloInfo{
		CipherSuites:      clientHello.cipherSuites,
		ServerName:        clientHello.serverName,
		SupportedCurves:   clientHello.supportedCurves,
		SupportedPoints:   clientHello.supportedPoints,
		SignatureSchemes:  clientHello.supportedSignatureAlgorithms,
		SupportedProtos:   clientHello.alpnProtocols,
		SupportedVersions: supportedVersions,
		Extensions:        clientHello.extensions,
		Conn:              c.conn,
		HelloRetryRequest: c.didHRR,
		config:            c.config,
		ctx:               ctx,
	}
}
