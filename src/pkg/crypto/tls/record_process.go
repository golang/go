// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

// A recordProcessor accepts reassembled records, decrypts and verifies them
// and routes them either to the handshake processor, to up to the application.
// It also accepts requests from the application for the current connection
// state, or for a notification when the state changes.

import (
	"container/list"
	"crypto/subtle"
	"hash"
)

// getConnectionState is a request from the application to get the current
// ConnectionState.
type getConnectionState struct {
	reply chan<- ConnectionState
}

// waitConnectionState is a request from the application to be notified when
// the connection state changes.
type waitConnectionState struct {
	reply chan<- ConnectionState
}

// connectionStateChange is a message from the handshake processor that the
// connection state has changed.
type connectionStateChange struct {
	connState ConnectionState
}

// changeCipherSpec is a message send to the handshake processor to signal that
// the peer is switching ciphers.
type changeCipherSpec struct{}

// newCipherSpec is a message from the handshake processor that future
// records should be processed with a new cipher and MAC function.
type newCipherSpec struct {
	encrypt encryptor
	mac     hash.Hash
}

type recordProcessor struct {
	decrypt       encryptor
	mac           hash.Hash
	seqNum        uint64
	handshakeBuf  []byte
	appDataChan   chan<- []byte
	requestChan   <-chan interface{}
	controlChan   <-chan interface{}
	recordChan    <-chan *record
	handshakeChan chan<- interface{}

	// recordRead is nil when we don't wish to read any more.
	recordRead <-chan *record
	// appDataSend is nil when len(appData) == 0.
	appDataSend chan<- []byte
	// appData contains any application data queued for upstream.
	appData []byte
	// A list of channels waiting for connState to change.
	waitQueue *list.List
	connState ConnectionState
	shutdown  bool
	header    [13]byte
}

// drainRequestChannel processes messages from the request channel until it's closed.
func drainRequestChannel(requestChan <-chan interface{}, c ConnectionState) {
	for v := range requestChan {
		if closed(requestChan) {
			return
		}
		switch r := v.(type) {
		case getConnectionState:
			r.reply <- c
		case waitConnectionState:
			r.reply <- c
		}
	}
}

func (p *recordProcessor) loop(appDataChan chan<- []byte, requestChan <-chan interface{}, controlChan <-chan interface{}, recordChan <-chan *record, handshakeChan chan<- interface{}) {
	noop := nop{}
	p.decrypt = noop
	p.mac = noop
	p.waitQueue = list.New()

	p.appDataChan = appDataChan
	p.requestChan = requestChan
	p.controlChan = controlChan
	p.recordChan = recordChan
	p.handshakeChan = handshakeChan
	p.recordRead = recordChan

	for !p.shutdown {
		select {
		case p.appDataSend <- p.appData:
			p.appData = nil
			p.appDataSend = nil
			p.recordRead = p.recordChan
		case c := <-controlChan:
			p.processControlMsg(c)
		case r := <-requestChan:
			p.processRequestMsg(r)
		case r := <-p.recordRead:
			p.processRecord(r)
		}
	}

	p.wakeWaiters()
	go drainRequestChannel(p.requestChan, p.connState)
	go func() {
		for _ = range controlChan {
		}
	}()

	close(handshakeChan)
	if len(p.appData) > 0 {
		appDataChan <- p.appData
	}
	close(appDataChan)
}

func (p *recordProcessor) processRequestMsg(requestMsg interface{}) {
	if closed(p.requestChan) {
		p.shutdown = true
		return
	}

	switch r := requestMsg.(type) {
	case getConnectionState:
		r.reply <- p.connState
	case waitConnectionState:
		if p.connState.HandshakeComplete {
			r.reply <- p.connState
		}
		p.waitQueue.PushBack(r.reply)
	}
}

func (p *recordProcessor) processControlMsg(msg interface{}) {
	connState, ok := msg.(ConnectionState)
	if !ok || closed(p.controlChan) {
		p.shutdown = true
		return
	}

	p.connState = connState
	p.wakeWaiters()
}

func (p *recordProcessor) wakeWaiters() {
	for i := p.waitQueue.Front(); i != nil; i = i.Next() {
		i.Value.(chan<- ConnectionState) <- p.connState
	}
	p.waitQueue.Init()
}

func (p *recordProcessor) processRecord(r *record) {
	if closed(p.recordChan) {
		p.shutdown = true
		return
	}

	p.decrypt.XORKeyStream(r.payload)
	if len(r.payload) < p.mac.Size() {
		p.error(alertBadRecordMAC)
		return
	}

	fillMACHeader(&p.header, p.seqNum, len(r.payload)-p.mac.Size(), r)
	p.seqNum++

	p.mac.Reset()
	p.mac.Write(p.header[0:13])
	p.mac.Write(r.payload[0 : len(r.payload)-p.mac.Size()])
	macBytes := p.mac.Sum()

	if subtle.ConstantTimeCompare(macBytes, r.payload[len(r.payload)-p.mac.Size():]) != 1 {
		p.error(alertBadRecordMAC)
		return
	}

	switch r.contentType {
	case recordTypeHandshake:
		p.processHandshakeRecord(r.payload[0 : len(r.payload)-p.mac.Size()])
	case recordTypeChangeCipherSpec:
		if len(r.payload) != 1 || r.payload[0] != 1 {
			p.error(alertUnexpectedMessage)
			return
		}

		p.handshakeChan <- changeCipherSpec{}
		newSpec, ok := (<-p.controlChan).(*newCipherSpec)
		if !ok {
			p.connState.Error = alertUnexpectedMessage
			p.shutdown = true
			return
		}
		p.decrypt = newSpec.encrypt
		p.mac = newSpec.mac
		p.seqNum = 0
	case recordTypeApplicationData:
		if p.connState.HandshakeComplete == false {
			p.error(alertUnexpectedMessage)
			return
		}
		p.recordRead = nil
		p.appData = r.payload[0 : len(r.payload)-p.mac.Size()]
		p.appDataSend = p.appDataChan
	default:
		p.error(alertUnexpectedMessage)
		return
	}
}

func (p *recordProcessor) processHandshakeRecord(data []byte) {
	if p.handshakeBuf == nil {
		p.handshakeBuf = data
	} else {
		if len(p.handshakeBuf) > maxHandshakeMsg {
			p.error(alertInternalError)
			return
		}
		newBuf := make([]byte, len(p.handshakeBuf)+len(data))
		copy(newBuf, p.handshakeBuf)
		copy(newBuf[len(p.handshakeBuf):], data)
		p.handshakeBuf = newBuf
	}

	for len(p.handshakeBuf) >= 4 {
		handshakeLen := int(p.handshakeBuf[1])<<16 |
			int(p.handshakeBuf[2])<<8 |
			int(p.handshakeBuf[3])
		if handshakeLen+4 > len(p.handshakeBuf) {
			break
		}

		bytes := p.handshakeBuf[0 : handshakeLen+4]
		p.handshakeBuf = p.handshakeBuf[handshakeLen+4:]
		if bytes[0] == typeFinished {
			// Special case because Finished is synchronous: the
			// handshake handler has to tell us if it's ok to start
			// forwarding application data.
			m := new(finishedMsg)
			if !m.unmarshal(bytes) {
				p.error(alertUnexpectedMessage)
			}
			p.handshakeChan <- m
			var ok bool
			p.connState, ok = (<-p.controlChan).(ConnectionState)
			if !ok || p.connState.Error != 0 {
				p.shutdown = true
				return
			}
		} else {
			msg, ok := parseHandshakeMsg(bytes)
			if !ok {
				p.error(alertUnexpectedMessage)
				return
			}
			p.handshakeChan <- msg
		}
	}
}

func (p *recordProcessor) error(err alertType) {
	close(p.handshakeChan)
	p.connState.Error = err
	p.wakeWaiters()
	p.shutdown = true
}

func parseHandshakeMsg(data []byte) (interface{}, bool) {
	var m interface {
		unmarshal([]byte) bool
	}

	switch data[0] {
	case typeClientHello:
		m = new(clientHelloMsg)
	case typeServerHello:
		m = new(serverHelloMsg)
	case typeCertificate:
		m = new(certificateMsg)
	case typeServerHelloDone:
		m = new(serverHelloDoneMsg)
	case typeClientKeyExchange:
		m = new(clientKeyExchangeMsg)
	case typeNextProtocol:
		m = new(nextProtoMsg)
	default:
		return nil, false
	}

	ok := m.unmarshal(data)
	return m, ok
}
