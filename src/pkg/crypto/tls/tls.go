// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package partially implements the TLS 1.1 protocol, as specified in RFC 4346.
package tls

import (
	"io"
	"os"
	"net"
	"time"
)

// A Conn represents a secure connection.
type Conn struct {
	net.Conn
	writeChan                 chan<- []byte
	readChan                  <-chan []byte
	requestChan               chan<- interface{}
	readBuf                   []byte
	eof                       bool
	readTimeout, writeTimeout int64
}

func timeout(c chan<- bool, nsecs int64) {
	time.Sleep(nsecs)
	c <- true
}

func (tls *Conn) Read(p []byte) (int, os.Error) {
	if len(tls.readBuf) == 0 {
		if tls.eof {
			return 0, os.EOF
		}

		var timeoutChan chan bool
		if tls.readTimeout > 0 {
			timeoutChan = make(chan bool)
			go timeout(timeoutChan, tls.readTimeout)
		}

		select {
		case b := <-tls.readChan:
			tls.readBuf = b
		case <-timeoutChan:
			return 0, os.EAGAIN
		}

		// TLS distinguishes between orderly closes and truncations. An
		// orderly close is represented by a zero length slice.
		if closed(tls.readChan) {
			return 0, io.ErrUnexpectedEOF
		}
		if len(tls.readBuf) == 0 {
			tls.eof = true
			return 0, os.EOF
		}
	}

	n := copy(p, tls.readBuf)
	tls.readBuf = tls.readBuf[n:]
	return n, nil
}

func (tls *Conn) Write(p []byte) (int, os.Error) {
	if tls.eof || closed(tls.readChan) {
		return 0, os.EOF
	}

	var timeoutChan chan bool
	if tls.writeTimeout > 0 {
		timeoutChan = make(chan bool)
		go timeout(timeoutChan, tls.writeTimeout)
	}

	select {
	case tls.writeChan <- p:
	case <-timeoutChan:
		return 0, os.EAGAIN
	}

	return len(p), nil
}

func (tls *Conn) Close() os.Error {
	close(tls.writeChan)
	close(tls.requestChan)
	tls.eof = true
	return nil
}

func (tls *Conn) SetTimeout(nsec int64) os.Error {
	tls.readTimeout = nsec
	tls.writeTimeout = nsec
	return nil
}

func (tls *Conn) SetReadTimeout(nsec int64) os.Error {
	tls.readTimeout = nsec
	return nil
}

func (tls *Conn) SetWriteTimeout(nsec int64) os.Error {
	tls.writeTimeout = nsec
	return nil
}

func (tls *Conn) GetConnectionState() ConnectionState {
	replyChan := make(chan ConnectionState)
	tls.requestChan <- getConnectionState{replyChan}
	return <-replyChan
}

func (tls *Conn) WaitConnectionState() ConnectionState {
	replyChan := make(chan ConnectionState)
	tls.requestChan <- waitConnectionState{replyChan}
	return <-replyChan
}

type handshaker interface {
	loop(writeChan chan<- interface{}, controlChan chan<- interface{}, msgChan <-chan interface{}, config *Config)
}

// Server establishes a secure connection over the given connection and acts
// as a TLS server.
func startTLSGoroutines(conn net.Conn, h handshaker, config *Config) *Conn {
	tls := new(Conn)
	tls.Conn = conn

	writeChan := make(chan []byte)
	readChan := make(chan []byte)
	requestChan := make(chan interface{})

	tls.writeChan = writeChan
	tls.readChan = readChan
	tls.requestChan = requestChan

	handshakeWriterChan := make(chan interface{})
	processorHandshakeChan := make(chan interface{})
	handshakeProcessorChan := make(chan interface{})
	readerProcessorChan := make(chan *record)

	go new(recordWriter).loop(conn, writeChan, handshakeWriterChan)
	go recordReader(readerProcessorChan, conn)
	go new(recordProcessor).loop(readChan, requestChan, handshakeProcessorChan, readerProcessorChan, processorHandshakeChan)
	go h.loop(handshakeWriterChan, handshakeProcessorChan, processorHandshakeChan, config)

	return tls
}

func Server(conn net.Conn, config *Config) *Conn {
	return startTLSGoroutines(conn, new(serverHandshake), config)
}

func Client(conn net.Conn, config *Config) *Conn {
	return startTLSGoroutines(conn, new(clientHandshake), config)
}

type Listener struct {
	listener net.Listener
	config   *Config
}

func (l Listener) Accept() (c net.Conn, err os.Error) {
	c, err = l.listener.Accept()
	if err != nil {
		return
	}

	c = Server(c, l.config)
	return
}

func (l Listener) Close() os.Error { return l.listener.Close() }

func (l Listener) Addr() net.Addr { return l.listener.Addr() }

// NewListener creates a Listener which accepts connections from an inner
// Listener and wraps each connection with Server.
func NewListener(listener net.Listener, config *Config) (l Listener) {
	l.listener = listener
	l.config = config
	return
}
