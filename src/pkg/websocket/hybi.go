// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

// This file implements a protocol of hybi draft.
// http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-17

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
)

const (
	websocketGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

	closeStatusNormal            = 1000
	closeStatusGoingAway         = 1001
	closeStatusProtocolError     = 1002
	closeStatusUnsupportedData   = 1003
	closeStatusFrameTooLarge     = 1004
	closeStatusNoStatusRcvd      = 1005
	closeStatusAbnormalClosure   = 1006
	closeStatusBadMessageData    = 1007
	closeStatusPolicyViolation   = 1008
	closeStatusTooBigData        = 1009
	closeStatusExtensionMismatch = 1010

	maxControlFramePayloadLength = 125
)

var (
	ErrBadMaskingKey         = &ProtocolError{"bad masking key"}
	ErrBadPongMessage        = &ProtocolError{"bad pong message"}
	ErrBadClosingStatus      = &ProtocolError{"bad closing status"}
	ErrUnsupportedExtensions = &ProtocolError{"unsupported extensions"}
	ErrNotImplemented        = &ProtocolError{"not implemented"}
)

// A hybiFrameHeader is a frame header as defined in hybi draft.
type hybiFrameHeader struct {
	Fin        bool
	Rsv        [3]bool
	OpCode     byte
	Length     int64
	MaskingKey []byte

	data *bytes.Buffer
}

// A hybiFrameReader is a reader for hybi frame.
type hybiFrameReader struct {
	reader io.Reader

	header hybiFrameHeader
	pos    int64
	length int
}

func (frame *hybiFrameReader) Read(msg []byte) (n int, err error) {
	n, err = frame.reader.Read(msg)
	if err != nil {
		return 0, err
	}
	if frame.header.MaskingKey != nil {
		for i := 0; i < n; i++ {
			msg[i] = msg[i] ^ frame.header.MaskingKey[frame.pos%4]
			frame.pos++
		}
	}
	return n, err
}

func (frame *hybiFrameReader) PayloadType() byte { return frame.header.OpCode }

func (frame *hybiFrameReader) HeaderReader() io.Reader {
	if frame.header.data == nil {
		return nil
	}
	if frame.header.data.Len() == 0 {
		return nil
	}
	return frame.header.data
}

func (frame *hybiFrameReader) TrailerReader() io.Reader { return nil }

func (frame *hybiFrameReader) Len() (n int) { return frame.length }

// A hybiFrameReaderFactory creates new frame reader based on its frame type.
type hybiFrameReaderFactory struct {
	*bufio.Reader
}

// NewFrameReader reads a frame header from the connection, and creates new reader for the frame.
// See Section 5.2 Base Frameing protocol for detail.
// http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-17#section-5.2
func (buf hybiFrameReaderFactory) NewFrameReader() (frame frameReader, err error) {
	hybiFrame := new(hybiFrameReader)
	frame = hybiFrame
	var header []byte
	var b byte
	// First byte. FIN/RSV1/RSV2/RSV3/OpCode(4bits)
	b, err = buf.ReadByte()
	if err != nil {
		return
	}
	header = append(header, b)
	hybiFrame.header.Fin = ((header[0] >> 7) & 1) != 0
	for i := 0; i < 3; i++ {
		j := uint(6 - i)
		hybiFrame.header.Rsv[i] = ((header[0] >> j) & 1) != 0
	}
	hybiFrame.header.OpCode = header[0] & 0x0f

	// Second byte. Mask/Payload len(7bits)
	b, err = buf.ReadByte()
	if err != nil {
		return
	}
	header = append(header, b)
	mask := (b & 0x80) != 0
	b &= 0x7f
	lengthFields := 0
	switch {
	case b <= 125: // Payload length 7bits.
		hybiFrame.header.Length = int64(b)
	case b == 126: // Payload length 7+16bits
		lengthFields = 2
	case b == 127: // Payload length 7+64bits
		lengthFields = 8
	}
	for i := 0; i < lengthFields; i++ {
		b, err = buf.ReadByte()
		if err != nil {
			return
		}
		header = append(header, b)
		hybiFrame.header.Length = hybiFrame.header.Length*256 + int64(b)
	}
	if mask {
		// Masking key. 4 bytes.
		for i := 0; i < 4; i++ {
			b, err = buf.ReadByte()
			if err != nil {
				return
			}
			header = append(header, b)
			hybiFrame.header.MaskingKey = append(hybiFrame.header.MaskingKey, b)
		}
	}
	hybiFrame.reader = io.LimitReader(buf.Reader, hybiFrame.header.Length)
	hybiFrame.header.data = bytes.NewBuffer(header)
	hybiFrame.length = len(header) + int(hybiFrame.header.Length)
	return
}

// A HybiFrameWriter is a writer for hybi frame.
type hybiFrameWriter struct {
	writer *bufio.Writer

	header *hybiFrameHeader
}

func (frame *hybiFrameWriter) Write(msg []byte) (n int, err error) {
	var header []byte
	var b byte
	if frame.header.Fin {
		b |= 0x80
	}
	for i := 0; i < 3; i++ {
		if frame.header.Rsv[i] {
			j := uint(6 - i)
			b |= 1 << j
		}
	}
	b |= frame.header.OpCode
	header = append(header, b)
	if frame.header.MaskingKey != nil {
		b = 0x80
	} else {
		b = 0
	}
	lengthFields := 0
	length := len(msg)
	switch {
	case length <= 125:
		b |= byte(length)
	case length < 65536:
		b |= 126
		lengthFields = 2
	default:
		b |= 127
		lengthFields = 8
	}
	header = append(header, b)
	for i := 0; i < lengthFields; i++ {
		j := uint((lengthFields - i - 1) * 8)
		b = byte((length >> j) & 0xff)
		header = append(header, b)
	}
	if frame.header.MaskingKey != nil {
		if len(frame.header.MaskingKey) != 4 {
			return 0, ErrBadMaskingKey
		}
		header = append(header, frame.header.MaskingKey...)
		frame.writer.Write(header)
		var data []byte

		for i := 0; i < length; i++ {
			data = append(data, msg[i]^frame.header.MaskingKey[i%4])
		}
		frame.writer.Write(data)
		err = frame.writer.Flush()
		return length, err
	}
	frame.writer.Write(header)
	frame.writer.Write(msg)
	err = frame.writer.Flush()
	return length, err
}

func (frame *hybiFrameWriter) Close() error { return nil }

type hybiFrameWriterFactory struct {
	*bufio.Writer
	needMaskingKey bool
}

func (buf hybiFrameWriterFactory) NewFrameWriter(payloadType byte) (frame frameWriter, err error) {
	frameHeader := &hybiFrameHeader{Fin: true, OpCode: payloadType}
	if buf.needMaskingKey {
		frameHeader.MaskingKey, err = generateMaskingKey()
		if err != nil {
			return nil, err
		}
	}
	return &hybiFrameWriter{writer: buf.Writer, header: frameHeader}, nil
}

type hybiFrameHandler struct {
	conn        *Conn
	payloadType byte
}

func (handler *hybiFrameHandler) HandleFrame(frame frameReader) (r frameReader, err error) {
	if handler.conn.IsServerConn() {
		// The client MUST mask all frames sent to the server.
		if frame.(*hybiFrameReader).header.MaskingKey == nil {
			handler.WriteClose(closeStatusProtocolError)
			return nil, io.EOF
		}
	} else {
		// The server MUST NOT mask all frames.
		if frame.(*hybiFrameReader).header.MaskingKey != nil {
			handler.WriteClose(closeStatusProtocolError)
			return nil, io.EOF
		}
	}
	if header := frame.HeaderReader(); header != nil {
		io.Copy(ioutil.Discard, header)
	}
	switch frame.PayloadType() {
	case ContinuationFrame:
		frame.(*hybiFrameReader).header.OpCode = handler.payloadType
	case TextFrame, BinaryFrame:
		handler.payloadType = frame.PayloadType()
	case CloseFrame:
		return nil, io.EOF
	case PingFrame:
		pingMsg := make([]byte, maxControlFramePayloadLength)
		n, err := io.ReadFull(frame, pingMsg)
		if err != nil && err != io.ErrUnexpectedEOF {
			return nil, err
		}
		io.Copy(ioutil.Discard, frame)
		n, err = handler.WritePong(pingMsg[:n])
		if err != nil {
			return nil, err
		}
		return nil, nil
	case PongFrame:
		return nil, ErrNotImplemented
	}
	return frame, nil
}

func (handler *hybiFrameHandler) WriteClose(status int) (err error) {
	handler.conn.wio.Lock()
	defer handler.conn.wio.Unlock()
	w, err := handler.conn.frameWriterFactory.NewFrameWriter(CloseFrame)
	if err != nil {
		return err
	}
	msg := make([]byte, 2)
	binary.BigEndian.PutUint16(msg, uint16(status))
	_, err = w.Write(msg)
	w.Close()
	return err
}

func (handler *hybiFrameHandler) WritePong(msg []byte) (n int, err error) {
	handler.conn.wio.Lock()
	defer handler.conn.wio.Unlock()
	w, err := handler.conn.frameWriterFactory.NewFrameWriter(PongFrame)
	if err != nil {
		return 0, err
	}
	n, err = w.Write(msg)
	w.Close()
	return n, err
}

// newHybiConn creates a new WebSocket connection speaking hybi draft protocol.
func newHybiConn(config *Config, buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) *Conn {
	if buf == nil {
		br := bufio.NewReader(rwc)
		bw := bufio.NewWriter(rwc)
		buf = bufio.NewReadWriter(br, bw)
	}
	ws := &Conn{config: config, request: request, buf: buf, rwc: rwc,
		frameReaderFactory: hybiFrameReaderFactory{buf.Reader},
		frameWriterFactory: hybiFrameWriterFactory{
			buf.Writer, request == nil},
		PayloadType:        TextFrame,
		defaultCloseStatus: closeStatusNormal}
	ws.frameHandler = &hybiFrameHandler{conn: ws}
	return ws
}

// generateMaskingKey generates a masking key for a frame.
func generateMaskingKey() (maskingKey []byte, err error) {
	maskingKey = make([]byte, 4)
	if _, err = io.ReadFull(rand.Reader, maskingKey); err != nil {
		return
	}
	return
}

// genetateNonce geneates a nonce consisting of a randomly selected 16-byte
// value that has been base64-encoded.
func generateNonce() (nonce []byte) {
	key := make([]byte, 16)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		panic(err)
	}
	nonce = make([]byte, 24)
	base64.StdEncoding.Encode(nonce, key)
	return
}

// getNonceAccept computes the base64-encoded SHA-1 of the concatenation of
// the nonce ("Sec-WebSocket-Key" value) with the websocket GUID string.
func getNonceAccept(nonce []byte) (expected []byte, err error) {
	h := sha1.New()
	if _, err = h.Write(nonce); err != nil {
		return
	}
	if _, err = h.Write([]byte(websocketGUID)); err != nil {
		return
	}
	expected = make([]byte, 28)
	base64.StdEncoding.Encode(expected, h.Sum(nil))
	return
}

func isHybiVersion(version int) bool {
	switch version {
	case ProtocolVersionHybi08, ProtocolVersionHybi13:
		return true
	default:
	}
	return false
}

// Client handhake described in draft-ietf-hybi-thewebsocket-protocol-17
func hybiClientHandshake(config *Config, br *bufio.Reader, bw *bufio.Writer) (err error) {
	if !isHybiVersion(config.Version) {
		panic("wrong protocol version.")
	}

	bw.WriteString("GET " + config.Location.RawPath + " HTTP/1.1\r\n")

	bw.WriteString("Host: " + config.Location.Host + "\r\n")
	bw.WriteString("Upgrade: websocket\r\n")
	bw.WriteString("Connection: Upgrade\r\n")
	nonce := generateNonce()
	if config.handshakeData != nil {
		nonce = []byte(config.handshakeData["key"])
	}
	bw.WriteString("Sec-WebSocket-Key: " + string(nonce) + "\r\n")
	if config.Version == ProtocolVersionHybi13 {
		bw.WriteString("Origin: " + strings.ToLower(config.Origin.String()) + "\r\n")
	} else if config.Version == ProtocolVersionHybi08 {
		bw.WriteString("Sec-WebSocket-Origin: " + strings.ToLower(config.Origin.String()) + "\r\n")
	}
	bw.WriteString("Sec-WebSocket-Version: " + fmt.Sprintf("%d", config.Version) + "\r\n")
	if len(config.Protocol) > 0 {
		bw.WriteString("Sec-WebSocket-Protocol: " + strings.Join(config.Protocol, ", ") + "\r\n")
	}
	// TODO(ukai): send extensions.
	// TODO(ukai): send cookie if any.

	bw.WriteString("\r\n")
	if err = bw.Flush(); err != nil {
		return err
	}

	resp, err := http.ReadResponse(br, &http.Request{Method: "GET"})
	if err != nil {
		return err
	}
	if resp.StatusCode != 101 {
		return ErrBadStatus
	}
	if strings.ToLower(resp.Header.Get("Upgrade")) != "websocket" ||
		strings.ToLower(resp.Header.Get("Connection")) != "upgrade" {
		return ErrBadUpgrade
	}
	expectedAccept, err := getNonceAccept(nonce)
	if err != nil {
		return err
	}
	if resp.Header.Get("Sec-WebSocket-Accept") != string(expectedAccept) {
		return ErrChallengeResponse
	}
	if resp.Header.Get("Sec-WebSocket-Extensions") != "" {
		return ErrUnsupportedExtensions
	}
	offeredProtocol := resp.Header.Get("Sec-WebSocket-Protocol")
	if offeredProtocol != "" {
		protocolMatched := false
		for i := 0; i < len(config.Protocol); i++ {
			if config.Protocol[i] == offeredProtocol {
				protocolMatched = true
				break
			}
		}
		if !protocolMatched {
			return ErrBadWebSocketProtocol
		}
		config.Protocol = []string{offeredProtocol}
	}

	return nil
}

// newHybiClientConn creates a client WebSocket connection after handshake.
func newHybiClientConn(config *Config, buf *bufio.ReadWriter, rwc io.ReadWriteCloser) *Conn {
	return newHybiConn(config, buf, rwc, nil)
}

// A HybiServerHandshaker performs a server handshake using hybi draft protocol.
type hybiServerHandshaker struct {
	*Config
	accept []byte
}

func (c *hybiServerHandshaker) ReadHandshake(buf *bufio.Reader, req *http.Request) (code int, err error) {
	c.Version = ProtocolVersionHybi13
	if req.Method != "GET" {
		return http.StatusMethodNotAllowed, ErrBadRequestMethod
	}
	// HTTP version can be safely ignored.

	if strings.ToLower(req.Header.Get("Upgrade")) != "websocket" ||
		!strings.Contains(strings.ToLower(req.Header.Get("Connection")), "upgrade") {
		return http.StatusBadRequest, ErrNotWebSocket
	}

	key := req.Header.Get("Sec-Websocket-Key")
	if key == "" {
		return http.StatusBadRequest, ErrChallengeResponse
	}
	version := req.Header.Get("Sec-Websocket-Version")
	var origin string
	switch version {
	case "13":
		c.Version = ProtocolVersionHybi13
		origin = req.Header.Get("Origin")
	case "8":
		c.Version = ProtocolVersionHybi08
		origin = req.Header.Get("Sec-Websocket-Origin")
	default:
		return http.StatusBadRequest, ErrBadWebSocketVersion
	}
	c.Origin, err = url.ParseRequest(origin)
	if err != nil {
		return http.StatusForbidden, err
	}
	var scheme string
	if req.TLS != nil {
		scheme = "wss"
	} else {
		scheme = "ws"
	}
	c.Location, err = url.ParseRequest(scheme + "://" + req.Host + req.URL.RawPath)
	if err != nil {
		return http.StatusBadRequest, err
	}
	protocol := strings.TrimSpace(req.Header.Get("Sec-Websocket-Protocol"))
	protocols := strings.Split(protocol, ",")
	for i := 0; i < len(protocols); i++ {
		c.Protocol = append(c.Protocol, strings.TrimSpace(protocols[i]))
	}
	c.accept, err = getNonceAccept([]byte(key))
	if err != nil {
		return http.StatusInternalServerError, err
	}
	return http.StatusSwitchingProtocols, nil
}

func (c *hybiServerHandshaker) AcceptHandshake(buf *bufio.Writer) (err error) {
	if len(c.Protocol) > 0 {
		if len(c.Protocol) != 1 {
			return ErrBadWebSocketProtocol
		}
	}
	buf.WriteString("HTTP/1.1 101 Switching Protocols\r\n")
	buf.WriteString("Upgrade: websocket\r\n")
	buf.WriteString("Connection: Upgrade\r\n")
	buf.WriteString("Sec-WebSocket-Accept: " + string(c.accept) + "\r\n")
	if len(c.Protocol) > 0 {
		buf.WriteString("Sec-WebSocket-Protocol: " + c.Protocol[0] + "\r\n")
	}
	// TODO(ukai): support extensions
	buf.WriteString("\r\n")
	return buf.Flush()
}

func (c *hybiServerHandshaker) NewServerConn(buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) *Conn {
	return newHybiServerConn(c.Config, buf, rwc, request)
}

// newHybiServerConn returns a new WebSocket connection speaking hybi draft protocol.
func newHybiServerConn(config *Config, buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) *Conn {
	return newHybiConn(config, buf, rwc, request)
}
