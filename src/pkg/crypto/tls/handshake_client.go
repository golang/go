// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/hmac";
	"crypto/rc4";
	"crypto/rsa";
	"crypto/sha1";
	"crypto/subtle";
	"crypto/x509";
	"io";
)

// A serverHandshake performs the server side of the TLS 1.1 handshake protocol.
type clientHandshake struct {
	writeChan	chan<- interface{};
	controlChan	chan<- interface{};
	msgChan		<-chan interface{};
	config		*Config;
}

func (h *clientHandshake) loop(writeChan chan<- interface{}, controlChan chan<- interface{}, msgChan <-chan interface{}, config *Config) {
	h.writeChan = writeChan;
	h.controlChan = controlChan;
	h.msgChan = msgChan;
	h.config = config;

	defer close(writeChan);
	defer close(controlChan);

	finishedHash := newFinishedHash();

	hello := &clientHelloMsg{
		major: defaultMajor,
		minor: defaultMinor,
		cipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA},
		compressionMethods: []uint8{compressionNone},
		random: make([]byte, 32),
	};

	currentTime := uint32(config.Time());
	hello.random[0] = byte(currentTime >> 24);
	hello.random[1] = byte(currentTime >> 16);
	hello.random[2] = byte(currentTime >> 8);
	hello.random[3] = byte(currentTime);
	_, err := io.ReadFull(config.Rand, hello.random[4:len(hello.random)]);
	if err != nil {
		h.error(alertInternalError);
		return;
	}

	finishedHash.Write(hello.marshal());
	writeChan <- writerSetVersion{defaultMajor, defaultMinor};
	writeChan <- hello;

	serverHello, ok := h.readHandshakeMsg().(*serverHelloMsg);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}
	finishedHash.Write(serverHello.marshal());
	major, minor, ok := mutualVersion(serverHello.major, serverHello.minor);
	if !ok {
		h.error(alertProtocolVersion);
		return;
	}

	writeChan <- writerSetVersion{major, minor};

	if serverHello.cipherSuite != TLS_RSA_WITH_RC4_128_SHA ||
		serverHello.compressionMethod != compressionNone {
		h.error(alertUnexpectedMessage);
		return;
	}

	certMsg, ok := h.readHandshakeMsg().(*certificateMsg);
	if !ok || len(certMsg.certificates) == 0 {
		h.error(alertUnexpectedMessage);
		return;
	}
	finishedHash.Write(certMsg.marshal());

	certs := make([]*x509.Certificate, len(certMsg.certificates));
	for i, asn1Data := range certMsg.certificates {
		cert, err := x509.ParseCertificate(asn1Data);
		if err != nil {
			h.error(alertBadCertificate);
			return;
		}
		certs[i] = cert;
	}

	// TODO(agl): do better validation of certs: max path length, name restrictions etc.
	for i := 1; i < len(certs); i++ {
		if certs[i-1].CheckSignatureFrom(certs[i]) != nil {
			h.error(alertBadCertificate);
			return;
		}
	}

	if config.RootCAs != nil {
		root := config.RootCAs.FindParent(certs[len(certs)-1]);
		if root == nil {
			h.error(alertBadCertificate);
			return;
		}
		if certs[len(certs)-1].CheckSignatureFrom(root) != nil {
			h.error(alertBadCertificate);
			return;
		}
	}

	pub, ok := certs[0].PublicKey.(*rsa.PublicKey);
	if !ok {
		h.error(alertUnsupportedCertificate);
		return;
	}

	shd, ok := h.readHandshakeMsg().(*serverHelloDoneMsg);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}
	finishedHash.Write(shd.marshal());

	ckx := new(clientKeyExchangeMsg);
	preMasterSecret := make([]byte, 48);
	// Note that the version number in the preMasterSecret must be the
	// version offered in the ClientHello.
	preMasterSecret[0] = defaultMajor;
	preMasterSecret[1] = defaultMinor;
	_, err = io.ReadFull(config.Rand, preMasterSecret[2:len(preMasterSecret)]);
	if err != nil {
		h.error(alertInternalError);
		return;
	}

	ckx.ciphertext, err = rsa.EncryptPKCS1v15(config.Rand, pub, preMasterSecret);
	if err != nil {
		h.error(alertInternalError);
		return;
	}

	finishedHash.Write(ckx.marshal());
	writeChan <- ckx;

	suite := cipherSuites[0];
	masterSecret, clientMAC, serverMAC, clientKey, serverKey :=
		keysFromPreMasterSecret11(preMasterSecret, hello.random, serverHello.random, suite.hashLength, suite.cipherKeyLength);

	cipher, _ := rc4.NewCipher(clientKey);
	writeChan <- writerChangeCipherSpec{cipher, hmac.New(sha1.New(), clientMAC)};

	finished := new(finishedMsg);
	finished.verifyData = finishedHash.clientSum(masterSecret);
	finishedHash.Write(finished.marshal());
	writeChan <- finished;

	// TODO(agl): this is cut-through mode which should probably be an option.
	writeChan <- writerEnableApplicationData{};

	_, ok = h.readHandshakeMsg().(changeCipherSpec);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}

	cipher2, _ := rc4.NewCipher(serverKey);
	controlChan <- &newCipherSpec{cipher2, hmac.New(sha1.New(), serverMAC)};

	serverFinished, ok := h.readHandshakeMsg().(*finishedMsg);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}

	verify := finishedHash.serverSum(masterSecret);
	if len(verify) != len(serverFinished.verifyData) ||
		subtle.ConstantTimeCompare(verify, serverFinished.verifyData) != 1 {
		h.error(alertHandshakeFailure);
		return;
	}

	controlChan <- ConnectionState{true, "TLS_RSA_WITH_RC4_128_SHA", 0};

	// This should just block forever.
	_ = h.readHandshakeMsg();
	h.error(alertUnexpectedMessage);
	return;
}

func (h *clientHandshake) readHandshakeMsg() interface{} {
	v := <-h.msgChan;
	if closed(h.msgChan) {
		// If the channel closed then the processor received an error
		// from the peer and we don't want to echo it back to them.
		h.msgChan = nil;
		return 0;
	}
	if _, ok := v.(alert); ok {
		// We got an alert from the processor. We forward to the writer
		// and shutdown.
		h.writeChan <- v;
		h.msgChan = nil;
		return 0;
	}
	return v;
}

func (h *clientHandshake) error(e alertType) {
	if h.msgChan != nil {
		// If we didn't get an error from the processor, then we need
		// to tell it about the error.
		go func() {
			for _ = range h.msgChan {
			}
		}();
		h.controlChan <- ConnectionState{false, "", e};
		close(h.controlChan);
		h.writeChan <- alert{alertLevelError, e};
	}
}
