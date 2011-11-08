// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package websocket implements a client and server for the WebSocket protocol.
// The protocol is defined at http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol
package websocket

import (
	"bufio"
	"crypto/tls"
	"encoding/json"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"sync"
)

const (
	ProtocolVersionHixie75   = -75
	ProtocolVersionHixie76   = -76
	ProtocolVersionHybi00    = 0
	ProtocolVersionHybi08    = 8
	ProtocolVersionHybi13    = 13
	ProtocolVersionHybi      = ProtocolVersionHybi13
	SupportedProtocolVersion = "13, 8"

	ContinuationFrame = 0
	TextFrame         = 1
	BinaryFrame       = 2
	CloseFrame        = 8
	PingFrame         = 9
	PongFrame         = 10
	UnknownFrame      = 255
)

// WebSocket protocol errors.
type ProtocolError struct {
	ErrorString string
}

func (err *ProtocolError) Error() string { return err.ErrorString }

var (
	ErrBadProtocolVersion   = &ProtocolError{"bad protocol version"}
	ErrBadScheme            = &ProtocolError{"bad scheme"}
	ErrBadStatus            = &ProtocolError{"bad status"}
	ErrBadUpgrade           = &ProtocolError{"missing or bad upgrade"}
	ErrBadWebSocketOrigin   = &ProtocolError{"missing or bad WebSocket-Origin"}
	ErrBadWebSocketLocation = &ProtocolError{"missing or bad WebSocket-Location"}
	ErrBadWebSocketProtocol = &ProtocolError{"missing or bad WebSocket-Protocol"}
	ErrBadWebSocketVersion  = &ProtocolError{"missing or bad WebSocket Version"}
	ErrChallengeResponse    = &ProtocolError{"mismatch challenge/response"}
	ErrBadFrame             = &ProtocolError{"bad frame"}
	ErrBadFrameBoundary     = &ProtocolError{"not on frame boundary"}
	ErrNotWebSocket         = &ProtocolError{"not websocket protocol"}
	ErrBadRequestMethod     = &ProtocolError{"bad method"}
	ErrNotSupported         = &ProtocolError{"not supported"}
)

// Addr is an implementation of net.Addr for WebSocket.
type Addr struct {
	*url.URL
}

// Network returns the network type for a WebSocket, "websocket".
func (addr *Addr) Network() string { return "websocket" }

// Config is a WebSocket configuration
type Config struct {
	// A WebSocket server address.
	Location *url.URL

	// A Websocket client origin.
	Origin *url.URL

	// WebSocket subprotocols.
	Protocol []string

	// WebSocket protocol version.
	Version int

	// TLS config for secure WebSocket (wss).
	TlsConfig *tls.Config

	handshakeData map[string]string
}

// serverHandshaker is an interface to handle WebSocket server side handshake.
type serverHandshaker interface {
	// ReadHandshake reads handshake request message from client.
	// Returns http response code and error if any.
	ReadHandshake(buf *bufio.Reader, req *http.Request) (code int, err error)

	// AcceptHandshake accepts the client handshake request and sends
	// handshake response back to client.
	AcceptHandshake(buf *bufio.Writer) (err error)

	// NewServerConn creates a new WebSocket connection.
	NewServerConn(buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) (conn *Conn)
}

// frameReader is an interface to read a WebSocket frame.
type frameReader interface {
	// Reader is to read payload of the frame.
	io.Reader

	// PayloadType returns payload type.
	PayloadType() byte

	// HeaderReader returns a reader to read header of the frame.
	HeaderReader() io.Reader

	// TrailerReader returns a reader to read trailer of the frame.
	// If it returns nil, there is no trailer in the frame.
	TrailerReader() io.Reader

	// Len returns total length of the frame, including header and trailer.
	Len() int
}

// frameReaderFactory is an interface to creates new frame reader.
type frameReaderFactory interface {
	NewFrameReader() (r frameReader, err error)
}

// frameWriter is an interface to write a WebSocket frame.
type frameWriter interface {
	// Writer is to write playload of the frame.
	io.WriteCloser
}

// frameWriterFactory is an interface to create new frame writer.
type frameWriterFactory interface {
	NewFrameWriter(payloadType byte) (w frameWriter, err error)
}

type frameHandler interface {
	HandleFrame(frame frameReader) (r frameReader, err error)
	WriteClose(status int) (err error)
}

// Conn represents a WebSocket connection.
type Conn struct {
	config  *Config
	request *http.Request

	buf *bufio.ReadWriter
	rwc io.ReadWriteCloser

	rio sync.Mutex
	frameReaderFactory
	frameReader

	wio sync.Mutex
	frameWriterFactory

	frameHandler
	PayloadType        byte
	defaultCloseStatus int
}

// Read implements the io.Reader interface:
// it reads data of a frame from the WebSocket connection.
// if msg is not large enough for the frame data, it fills the msg and next Read
// will read the rest of the frame data.
// it reads Text frame or Binary frame.
func (ws *Conn) Read(msg []byte) (n int, err error) {
	ws.rio.Lock()
	defer ws.rio.Unlock()
again:
	if ws.frameReader == nil {
		frame, err := ws.frameReaderFactory.NewFrameReader()
		if err != nil {
			return 0, err
		}
		ws.frameReader, err = ws.frameHandler.HandleFrame(frame)
		if err != nil {
			return 0, err
		}
		if ws.frameReader == nil {
			goto again
		}
	}
	n, err = ws.frameReader.Read(msg)
	if err == io.EOF {
		if trailer := ws.frameReader.TrailerReader(); trailer != nil {
			io.Copy(ioutil.Discard, trailer)
		}
		ws.frameReader = nil
		goto again
	}
	return n, err
}

// Write implements the io.Writer interface:
// it writes data as a frame to the WebSocket connection.
func (ws *Conn) Write(msg []byte) (n int, err error) {
	ws.wio.Lock()
	defer ws.wio.Unlock()
	w, err := ws.frameWriterFactory.NewFrameWriter(ws.PayloadType)
	if err != nil {
		return 0, err
	}
	n, err = w.Write(msg)
	w.Close()
	if err != nil {
		return n, err
	}
	return n, err
}

// Close implements the io.Closer interface.
func (ws *Conn) Close() error {
	err := ws.frameHandler.WriteClose(ws.defaultCloseStatus)
	if err != nil {
		return err
	}
	return ws.rwc.Close()
}

func (ws *Conn) IsClientConn() bool { return ws.request == nil }
func (ws *Conn) IsServerConn() bool { return ws.request != nil }

// LocalAddr returns the WebSocket Origin for the connection for client, or
// the WebSocket location for server.
func (ws *Conn) LocalAddr() net.Addr {
	if ws.IsClientConn() {
		return &Addr{ws.config.Origin}
	}
	return &Addr{ws.config.Location}
}

// RemoteAddr returns the WebSocket location for the connection for client, or
// the Websocket Origin for server.
func (ws *Conn) RemoteAddr() net.Addr {
	if ws.IsClientConn() {
		return &Addr{ws.config.Location}
	}
	return &Addr{ws.config.Origin}
}

// SetTimeout sets the connection's network timeout in nanoseconds.
func (ws *Conn) SetTimeout(nsec int64) error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetTimeout(nsec)
	}
	return os.EINVAL
}

// SetReadTimeout sets the connection's network read timeout in nanoseconds.
func (ws *Conn) SetReadTimeout(nsec int64) error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetReadTimeout(nsec)
	}
	return os.EINVAL
}

// SetWriteTimeout sets the connection's network write timeout in nanoseconds.
func (ws *Conn) SetWriteTimeout(nsec int64) error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetWriteTimeout(nsec)
	}
	return os.EINVAL
}

// Config returns the WebSocket config.
func (ws *Conn) Config() *Config { return ws.config }

// Request returns the http request upgraded to the WebSocket.
// It is nil for client side.
func (ws *Conn) Request() *http.Request { return ws.request }

// Codec represents a symmetric pair of functions that implement a codec.
type Codec struct {
	Marshal   func(v interface{}) (data []byte, payloadType byte, err error)
	Unmarshal func(data []byte, payloadType byte, v interface{}) (err error)
}

// Send sends v marshaled by cd.Marshal as single frame to ws.
func (cd Codec) Send(ws *Conn, v interface{}) (err error) {
	if err != nil {
		return err
	}
	data, payloadType, err := cd.Marshal(v)
	if err != nil {
		return err
	}
	ws.wio.Lock()
	defer ws.wio.Unlock()
	w, err := ws.frameWriterFactory.NewFrameWriter(payloadType)
	_, err = w.Write(data)
	w.Close()
	return err
}

// Receive receives single frame from ws, unmarshaled by cd.Unmarshal and stores in v.
func (cd Codec) Receive(ws *Conn, v interface{}) (err error) {
	ws.rio.Lock()
	defer ws.rio.Unlock()
	if ws.frameReader != nil {
		_, err = io.Copy(ioutil.Discard, ws.frameReader)
		if err != nil {
			return err
		}
		ws.frameReader = nil
	}
again:
	frame, err := ws.frameReaderFactory.NewFrameReader()
	if err != nil {
		return err
	}
	frame, err = ws.frameHandler.HandleFrame(frame)
	if err != nil {
		return err
	}
	if frame == nil {
		goto again
	}
	payloadType := frame.PayloadType()
	data, err := ioutil.ReadAll(frame)
	if err != nil {
		return err
	}
	return cd.Unmarshal(data, payloadType, v)
}

func marshal(v interface{}) (msg []byte, payloadType byte, err error) {
	switch data := v.(type) {
	case string:
		return []byte(data), TextFrame, nil
	case []byte:
		return data, BinaryFrame, nil
	}
	return nil, UnknownFrame, ErrNotSupported
}

func unmarshal(msg []byte, payloadType byte, v interface{}) (err error) {
	switch data := v.(type) {
	case *string:
		*data = string(msg)
		return nil
	case *[]byte:
		*data = msg
		return nil
	}
	return ErrNotSupported
}

/*
Message is a codec to send/receive text/binary data in a frame on WebSocket connection.
To send/receive text frame, use string type.
To send/receive binary frame, use []byte type.

Trivial usage:

	import "websocket"

	// receive text frame
	var message string
	websocket.Message.Receive(ws, &message)

	// send text frame
	message = "hello"
	websocket.Message.Send(ws, message)

	// receive binary frame
	var data []byte
	websocket.Message.Receive(ws, &data)

	// send binary frame
	data = []byte{0, 1, 2}
	websocket.Message.Send(ws, data)

*/
var Message = Codec{marshal, unmarshal}

func jsonMarshal(v interface{}) (msg []byte, payloadType byte, err error) {
	msg, err = json.Marshal(v)
	return msg, TextFrame, err
}

func jsonUnmarshal(msg []byte, payloadType byte, v interface{}) (err error) {
	return json.Unmarshal(msg, v)
}

/*
JSON is a codec to send/receive JSON data in a frame from a WebSocket connection.

Trival usage:

	import "websocket"

	type T struct {
		Msg string
		Count int
	}

	// receive JSON type T
	var data T
	websocket.JSON.Receive(ws, &data)

	// send JSON type T
	websocket.JSON.Send(ws, data)
*/
var JSON = Codec{jsonMarshal, jsonUnmarshal}
