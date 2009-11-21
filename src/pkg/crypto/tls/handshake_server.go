// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

// The handshake goroutine reads handshake messages from the record processor
// and outputs messages to be written on another channel. It updates the record
// processor with the state of the connection via the control channel. In the
// case of handshake messages that need synchronous processing (because they
// affect the handling of the next record) the record processor knows about
// them and either waits for a control message (Finished) or includes a reply
// channel in the message (ChangeCipherSpec).

import (
	"crypto/hmac";
	"crypto/rc4";
	"crypto/rsa";
	"crypto/sha1";
	"crypto/subtle";
	"io";
)

type cipherSuite struct {
	id				uint16;	// The number of this suite on the wire.
	hashLength, cipherKeyLength	int;
	// TODO(agl): need a method to create the cipher and hash interfaces.
}

var cipherSuites = []cipherSuite{
	cipherSuite{TLS_RSA_WITH_RC4_128_SHA, 20, 16},
}

// A serverHandshake performs the server side of the TLS 1.1 handshake protocol.
type serverHandshake struct {
	writeChan	chan<- interface{};
	controlChan	chan<- interface{};
	msgChan		<-chan interface{};
	config		*Config;
}

func (h *serverHandshake) loop(writeChan chan<- interface{}, controlChan chan<- interface{}, msgChan <-chan interface{}, config *Config) {
	h.writeChan = writeChan;
	h.controlChan = controlChan;
	h.msgChan = msgChan;
	h.config = config;

	defer close(writeChan);
	defer close(controlChan);

	clientHello, ok := h.readHandshakeMsg().(*clientHelloMsg);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}
	major, minor, ok := mutualVersion(clientHello.major, clientHello.minor);
	if !ok {
		h.error(alertProtocolVersion);
		return;
	}

	finishedHash := newFinishedHash();
	finishedHash.Write(clientHello.marshal());

	hello := new(serverHelloMsg);

	// We only support a single ciphersuite so we look for it in the list
	// of client supported suites.
	//
	// TODO(agl): Add additional cipher suites.
	var suite *cipherSuite;

	for _, id := range clientHello.cipherSuites {
		for _, supported := range cipherSuites {
			if supported.id == id {
				suite = &supported;
				break;
			}
		}
	}

	foundCompression := false;
	// We only support null compression, so check that the client offered it.
	for _, compression := range clientHello.compressionMethods {
		if compression == compressionNone {
			foundCompression = true;
			break;
		}
	}

	if suite == nil || !foundCompression {
		h.error(alertHandshakeFailure);
		return;
	}

	hello.major = major;
	hello.minor = minor;
	hello.cipherSuite = suite.id;
	currentTime := uint32(config.Time());
	hello.random = make([]byte, 32);
	hello.random[0] = byte(currentTime >> 24);
	hello.random[1] = byte(currentTime >> 16);
	hello.random[2] = byte(currentTime >> 8);
	hello.random[3] = byte(currentTime);
	_, err := io.ReadFull(config.Rand, hello.random[4:]);
	if err != nil {
		h.error(alertInternalError);
		return;
	}
	hello.compressionMethod = compressionNone;

	finishedHash.Write(hello.marshal());
	writeChan <- writerSetVersion{major, minor};
	writeChan <- hello;

	if len(config.Certificates) == 0 {
		h.error(alertInternalError);
		return;
	}

	certMsg := new(certificateMsg);
	certMsg.certificates = config.Certificates[0].Certificate;
	finishedHash.Write(certMsg.marshal());
	writeChan <- certMsg;

	helloDone := new(serverHelloDoneMsg);
	finishedHash.Write(helloDone.marshal());
	writeChan <- helloDone;

	ckx, ok := h.readHandshakeMsg().(*clientKeyExchangeMsg);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}
	finishedHash.Write(ckx.marshal());

	preMasterSecret := make([]byte, 48);
	_, err = io.ReadFull(config.Rand, preMasterSecret[2:]);
	if err != nil {
		h.error(alertInternalError);
		return;
	}

	err = rsa.DecryptPKCS1v15SessionKey(config.Rand, config.Certificates[0].PrivateKey, ckx.ciphertext, preMasterSecret);
	if err != nil {
		h.error(alertHandshakeFailure);
		return;
	}
	// We don't check the version number in the premaster secret. For one,
	// by checking it, we would leak information about the validity of the
	// encrypted pre-master secret. Secondly, it provides only a small
	// benefit against a downgrade attack and some implementations send the
	// wrong version anyway. See the discussion at the end of section
	// 7.4.7.1 of RFC 4346.

	masterSecret, clientMAC, serverMAC, clientKey, serverKey :=
		keysFromPreMasterSecret11(preMasterSecret, clientHello.random, hello.random, suite.hashLength, suite.cipherKeyLength);

	_, ok = h.readHandshakeMsg().(changeCipherSpec);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}

	cipher, _ := rc4.NewCipher(clientKey);
	controlChan <- &newCipherSpec{cipher, hmac.New(sha1.New(), clientMAC)};

	clientFinished, ok := h.readHandshakeMsg().(*finishedMsg);
	if !ok {
		h.error(alertUnexpectedMessage);
		return;
	}

	verify := finishedHash.clientSum(masterSecret);
	if len(verify) != len(clientFinished.verifyData) ||
		subtle.ConstantTimeCompare(verify, clientFinished.verifyData) != 1 {
		h.error(alertHandshakeFailure);
		return;
	}

	controlChan <- ConnectionState{true, "TLS_RSA_WITH_RC4_128_SHA", 0};

	finishedHash.Write(clientFinished.marshal());

	cipher2, _ := rc4.NewCipher(serverKey);
	writeChan <- writerChangeCipherSpec{cipher2, hmac.New(sha1.New(), serverMAC)};

	finished := new(finishedMsg);
	finished.verifyData = finishedHash.serverSum(masterSecret);
	writeChan <- finished;

	writeChan <- writerEnableApplicationData{};

	for {
		_, ok := h.readHandshakeMsg().(*clientHelloMsg);
		if !ok {
			h.error(alertUnexpectedMessage);
			return;
		}
		// We reject all renegotication requests.
		writeChan <- alert{alertLevelWarning, alertNoRenegotiation};
	}
}

func (h *serverHandshake) readHandshakeMsg() interface{} {
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

func (h *serverHandshake) error(e alertType) {
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
