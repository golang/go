// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

// This file implements a protocol of Hixie draft version 75 and 76
// (draft 76 equals to hybi 00)

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// An aray of characters to be randomly inserted to construct Sec-WebSocket-Key
// value. It holds characters from ranges U+0021 to U+002F and U+003A to U+007E.
// See Step 21 in Section 4.1 Opening handshake.
// http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-00#page-22
var secKeyRandomChars [0x30 - 0x21 + 0x7F - 0x3A]byte

func init() {
	i := 0
	for ch := byte(0x21); ch < 0x30; ch++ {
		secKeyRandomChars[i] = ch
		i++
	}
	for ch := byte(0x3a); ch < 0x7F; ch++ {
		secKeyRandomChars[i] = ch
		i++
	}
}

type byteReader interface {
	ReadByte() (byte, error)
}

// readHixieLength reads frame length for frame type 0x80-0xFF
// as defined in Hixie draft.
// See section 4.2 Data framing.
// http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-00#section-4.2
func readHixieLength(r byteReader) (length int64, lengthFields []byte, err error) {
	for {
		c, err := r.ReadByte()
		if err != nil {
			return 0, nil, err
		}
		lengthFields = append(lengthFields, c)
		length = length*128 + int64(c&0x7f)
		if c&0x80 == 0 {
			break
		}
	}
	return
}

// A hixieLengthFrameReader is a reader for frame type 0x80-0xFF
// as defined in hixie draft.
type hixieLengthFrameReader struct {
	reader    io.Reader
	FrameType byte
	Length    int64
	header    *bytes.Buffer
	length    int
}

func (frame *hixieLengthFrameReader) Read(msg []byte) (n int, err error) {
	return frame.reader.Read(msg)
}

func (frame *hixieLengthFrameReader) PayloadType() byte {
	if frame.FrameType == '\xff' && frame.Length == 0 {
		return CloseFrame
	}
	return UnknownFrame
}

func (frame *hixieLengthFrameReader) HeaderReader() io.Reader {
	if frame.header == nil {
		return nil
	}
	if frame.header.Len() == 0 {
		frame.header = nil
		return nil
	}
	return frame.header
}

func (frame *hixieLengthFrameReader) TrailerReader() io.Reader { return nil }

func (frame *hixieLengthFrameReader) Len() (n int) { return frame.length }

// A HixieSentinelFrameReader is a reader for frame type 0x00-0x7F
// as defined in hixie draft.
type hixieSentinelFrameReader struct {
	reader      *bufio.Reader
	FrameType   byte
	header      *bytes.Buffer
	data        []byte
	seenTrailer bool
	trailer     *bytes.Buffer
}

func (frame *hixieSentinelFrameReader) Read(msg []byte) (n int, err error) {
	if len(frame.data) == 0 {
		if frame.seenTrailer {
			return 0, io.EOF
		}
		frame.data, err = frame.reader.ReadSlice('\xff')
		if err == nil {
			frame.seenTrailer = true
			frame.data = frame.data[:len(frame.data)-1] // trim \xff
			frame.trailer = bytes.NewBuffer([]byte{0xff})
		}
	}
	n = copy(msg, frame.data)
	frame.data = frame.data[n:]
	return n, err
}

func (frame *hixieSentinelFrameReader) PayloadType() byte {
	if frame.FrameType == 0 {
		return TextFrame
	}
	return UnknownFrame
}

func (frame *hixieSentinelFrameReader) HeaderReader() io.Reader {
	if frame.header == nil {
		return nil
	}
	if frame.header.Len() == 0 {
		frame.header = nil
		return nil
	}
	return frame.header
}

func (frame *hixieSentinelFrameReader) TrailerReader() io.Reader {
	if frame.trailer == nil {
		return nil
	}
	if frame.trailer.Len() == 0 {
		frame.trailer = nil
		return nil
	}
	return frame.trailer
}

func (frame *hixieSentinelFrameReader) Len() int { return -1 }

// A HixieFrameReaderFactory creates new frame reader based on its frame type.
type hixieFrameReaderFactory struct {
	*bufio.Reader
}

func (buf hixieFrameReaderFactory) NewFrameReader() (r frameReader, err error) {
	var header []byte
	var b byte
	b, err = buf.ReadByte()
	if err != nil {
		return
	}
	header = append(header, b)
	if b&0x80 == 0x80 {
		length, lengthFields, err := readHixieLength(buf.Reader)
		if err != nil {
			return nil, err
		}
		if length == 0 {
			return nil, io.EOF
		}
		header = append(header, lengthFields...)
		return &hixieLengthFrameReader{
			reader:    io.LimitReader(buf.Reader, length),
			FrameType: b,
			Length:    length,
			header:    bytes.NewBuffer(header)}, err
	}
	return &hixieSentinelFrameReader{
		reader:    buf.Reader,
		FrameType: b,
		header:    bytes.NewBuffer(header)}, err
}

type hixiFrameWriter struct {
	writer *bufio.Writer
}

func (frame *hixiFrameWriter) Write(msg []byte) (n int, err error) {
	frame.writer.WriteByte(0)
	frame.writer.Write(msg)
	frame.writer.WriteByte(0xff)
	err = frame.writer.Flush()
	return len(msg), err
}

func (frame *hixiFrameWriter) Close() error { return nil }

type hixiFrameWriterFactory struct {
	*bufio.Writer
}

func (buf hixiFrameWriterFactory) NewFrameWriter(payloadType byte) (frame frameWriter, err error) {
	if payloadType != TextFrame {
		return nil, ErrNotSupported
	}
	return &hixiFrameWriter{writer: buf.Writer}, nil
}

type hixiFrameHandler struct {
	conn *Conn
}

func (handler *hixiFrameHandler) HandleFrame(frame frameReader) (r frameReader, err error) {
	if header := frame.HeaderReader(); header != nil {
		io.Copy(ioutil.Discard, header)
	}
	if frame.PayloadType() != TextFrame {
		io.Copy(ioutil.Discard, frame)
		return nil, nil
	}
	return frame, nil
}

func (handler *hixiFrameHandler) WriteClose(_ int) (err error) {
	handler.conn.wio.Lock()
	defer handler.conn.wio.Unlock()
	closingFrame := []byte{'\xff', '\x00'}
	handler.conn.buf.Write(closingFrame)
	return handler.conn.buf.Flush()
}

// newHixiConn creates a new WebSocket connection speaking hixie draft protocol.
func newHixieConn(config *Config, buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) *Conn {
	if buf == nil {
		br := bufio.NewReader(rwc)
		bw := bufio.NewWriter(rwc)
		buf = bufio.NewReadWriter(br, bw)
	}
	ws := &Conn{config: config, request: request, buf: buf, rwc: rwc,
		frameReaderFactory: hixieFrameReaderFactory{buf.Reader},
		frameWriterFactory: hixiFrameWriterFactory{buf.Writer},
		PayloadType:        TextFrame}
	ws.frameHandler = &hixiFrameHandler{ws}
	return ws
}

// getChallengeResponse computes the expected response from the
// challenge as described in section 5.1 Opening Handshake steps 42 to
// 43 of http://www.whatwg.org/specs/web-socket-protocol/
func getChallengeResponse(number1, number2 uint32, key3 []byte) (expected []byte, err error) {
	// 41. Let /challenge/ be the concatenation of /number_1/, expressed
	// a big-endian 32 bit integer, /number_2/, expressed in a big-
	// endian 32 bit integer, and the eight bytes of /key_3/ in the
	// order they were sent to the wire.
	challenge := make([]byte, 16)
	binary.BigEndian.PutUint32(challenge[0:], number1)
	binary.BigEndian.PutUint32(challenge[4:], number2)
	copy(challenge[8:], key3)

	// 42. Let /expected/ be the MD5 fingerprint of /challenge/ as a big-
	// endian 128 bit string.
	h := md5.New()
	if _, err = h.Write(challenge); err != nil {
		return
	}
	expected = h.Sum(nil)
	return
}

// Generates handshake key as described in 4.1 Opening handshake step 16 to 22.
// cf. http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-00
func generateKeyNumber() (key string, number uint32) {
	// 16.  Let /spaces_n/ be a random integer from 1 to 12 inclusive.
	spaces := rand.Intn(12) + 1

	// 17. Let /max_n/ be the largest integer not greater than
	//     4,294,967,295 divided by /spaces_n/
	max := int(4294967295 / uint32(spaces))

	// 18. Let /number_n/ be a random integer from 0 to /max_n/ inclusive.
	number = uint32(rand.Intn(max + 1))

	// 19. Let /product_n/ be the result of multiplying /number_n/ and
	//     /spaces_n/ together.
	product := number * uint32(spaces)

	// 20. Let /key_n/ be a string consisting of /product_n/, expressed
	// in base ten using the numerals in the range U+0030 DIGIT ZERO (0)
	// to U+0039 DIGIT NINE (9).
	key = fmt.Sprintf("%d", product)

	// 21. Insert between one and twelve random characters from the ranges
	//     U+0021 to U+002F and U+003A to U+007E into /key_n/ at random
	//     positions.
	n := rand.Intn(12) + 1
	for i := 0; i < n; i++ {
		pos := rand.Intn(len(key)) + 1
		ch := secKeyRandomChars[rand.Intn(len(secKeyRandomChars))]
		key = key[0:pos] + string(ch) + key[pos:]
	}

	// 22. Insert /spaces_n/ U+0020 SPACE characters into /key_n/ at random
	//     positions other than the start or end of the string.
	for i := 0; i < spaces; i++ {
		pos := rand.Intn(len(key)-1) + 1
		key = key[0:pos] + " " + key[pos:]
	}

	return
}

// Generates handshake key_3 as described in 4.1 Opening handshake step 26.
// cf. http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-00
func generateKey3() (key []byte) {
	// 26. Let /key3/ be a string consisting of eight random bytes (or
	//  equivalently, a random 64 bit integer encoded in big-endian order).
	key = make([]byte, 8)
	for i := 0; i < 8; i++ {
		key[i] = byte(rand.Intn(256))
	}
	return
}

// Cilent handhake described in (soon obsolete)
// draft-ietf-hybi-thewebsocket-protocol-00
// (draft-hixie-thewebsocket-protocol-76) 
func hixie76ClientHandshake(config *Config, br *bufio.Reader, bw *bufio.Writer) (err error) {
	switch config.Version {
	case ProtocolVersionHixie76, ProtocolVersionHybi00:
	default:
		panic("wrong protocol version.")
	}
	// 4.1. Opening handshake.
	// Step 5.  send a request line.
	bw.WriteString("GET " + config.Location.RawPath + " HTTP/1.1\r\n")

	// Step 6-14. push request headers in fields.
	fields := []string{
		"Upgrade: WebSocket\r\n",
		"Connection: Upgrade\r\n",
		"Host: " + config.Location.Host + "\r\n",
		"Origin: " + config.Origin.String() + "\r\n",
	}
	if len(config.Protocol) > 0 {
		if len(config.Protocol) != 1 {
			return ErrBadWebSocketProtocol
		}
		fields = append(fields, "Sec-WebSocket-Protocol: "+config.Protocol[0]+"\r\n")
	}
	// TODO(ukai): Step 15. send cookie if any.

	// Step 16-23. generate keys and push Sec-WebSocket-Key<n> in fields.
	key1, number1 := generateKeyNumber()
	key2, number2 := generateKeyNumber()
	if config.handshakeData != nil {
		key1 = config.handshakeData["key1"]
		n, err := strconv.ParseUint(config.handshakeData["number1"], 10, 32)
		if err != nil {
			panic(err)
		}
		number1 = uint32(n)
		key2 = config.handshakeData["key2"]
		n, err = strconv.ParseUint(config.handshakeData["number2"], 10, 32)
		if err != nil {
			panic(err)
		}
		number2 = uint32(n)
	}
	fields = append(fields, "Sec-WebSocket-Key1: "+key1+"\r\n")
	fields = append(fields, "Sec-WebSocket-Key2: "+key2+"\r\n")

	// Step 24. shuffle fields and send them out.
	for i := 1; i < len(fields); i++ {
		j := rand.Intn(i)
		fields[i], fields[j] = fields[j], fields[i]
	}
	for i := 0; i < len(fields); i++ {
		bw.WriteString(fields[i])
	}
	// Step 25. send CRLF.
	bw.WriteString("\r\n")

	// Step 26. generate 8 bytes random key.
	key3 := generateKey3()
	if config.handshakeData != nil {
		key3 = []byte(config.handshakeData["key3"])
	}
	// Step 27. send it out.
	bw.Write(key3)
	if err = bw.Flush(); err != nil {
		return
	}

	// Step 28-29, 32-40. read response from server.
	resp, err := http.ReadResponse(br, &http.Request{Method: "GET"})
	if err != nil {
		return err
	}
	// Step 30. check response code is 101.
	if resp.StatusCode != 101 {
		return ErrBadStatus
	}

	// Step 41. check websocket headers.
	if resp.Header.Get("Upgrade") != "WebSocket" ||
		strings.ToLower(resp.Header.Get("Connection")) != "upgrade" {
		return ErrBadUpgrade
	}

	if resp.Header.Get("Sec-Websocket-Origin") != config.Origin.String() {
		return ErrBadWebSocketOrigin
	}

	if resp.Header.Get("Sec-Websocket-Location") != config.Location.String() {
		return ErrBadWebSocketLocation
	}

	if len(config.Protocol) > 0 && resp.Header.Get("Sec-Websocket-Protocol") != config.Protocol[0] {
		return ErrBadWebSocketProtocol
	}

	// Step 42-43. get expected data from challenge data.
	expected, err := getChallengeResponse(number1, number2, key3)
	if err != nil {
		return err
	}

	// Step 44. read 16 bytes from server.
	reply := make([]byte, 16)
	if _, err = io.ReadFull(br, reply); err != nil {
		return err
	}

	// Step 45. check the reply equals to expected data.
	if !bytes.Equal(expected, reply) {
		return ErrChallengeResponse
	}
	// WebSocket connection is established.
	return
}

// Client Handshake described in (soon obsolete)
// draft-hixie-thewebsocket-protocol-75.
func hixie75ClientHandshake(config *Config, br *bufio.Reader, bw *bufio.Writer) (err error) {
	if config.Version != ProtocolVersionHixie75 {
		panic("wrong protocol version.")
	}
	bw.WriteString("GET " + config.Location.RawPath + " HTTP/1.1\r\n")
	bw.WriteString("Upgrade: WebSocket\r\n")
	bw.WriteString("Connection: Upgrade\r\n")
	bw.WriteString("Host: " + config.Location.Host + "\r\n")
	bw.WriteString("Origin: " + config.Origin.String() + "\r\n")
	if len(config.Protocol) > 0 {
		if len(config.Protocol) != 1 {
			return ErrBadWebSocketProtocol
		}
		bw.WriteString("WebSocket-Protocol: " + config.Protocol[0] + "\r\n")
	}
	bw.WriteString("\r\n")
	bw.Flush()
	resp, err := http.ReadResponse(br, &http.Request{Method: "GET"})
	if err != nil {
		return
	}
	if resp.Status != "101 Web Socket Protocol Handshake" {
		return ErrBadStatus
	}
	if resp.Header.Get("Upgrade") != "WebSocket" ||
		resp.Header.Get("Connection") != "Upgrade" {
		return ErrBadUpgrade
	}
	if resp.Header.Get("Websocket-Origin") != config.Origin.String() {
		return ErrBadWebSocketOrigin
	}
	if resp.Header.Get("Websocket-Location") != config.Location.String() {
		return ErrBadWebSocketLocation
	}
	if len(config.Protocol) > 0 && resp.Header.Get("Websocket-Protocol") != config.Protocol[0] {
		return ErrBadWebSocketProtocol
	}
	return
}

// newHixieClientConn returns new WebSocket connection speaking hixie draft protocol.
func newHixieClientConn(config *Config, buf *bufio.ReadWriter, rwc io.ReadWriteCloser) *Conn {
	return newHixieConn(config, buf, rwc, nil)
}

// Gets key number from Sec-WebSocket-Key<n>: field as described
// in 5.2 Sending the server's opening handshake, 4.
func getKeyNumber(s string) (r uint32) {
	// 4. Let /key-number_n/ be the digits (characters in the range
	// U+0030 DIGIT ZERO (0) to U+0039 DIGIT NINE (9)) in /key_1/,
	// interpreted as a base ten integer, ignoring all other characters
	// in /key_n/.
	r = 0
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			r = r*10 + uint32(s[i]) - '0'
		}
	}
	return
}

// A Hixie76ServerHandshaker performs a server handshake using
// hixie draft 76 protocol.
type hixie76ServerHandshaker struct {
	*Config
	challengeResponse []byte
}

func (c *hixie76ServerHandshaker) ReadHandshake(buf *bufio.Reader, req *http.Request) (code int, err error) {
	c.Version = ProtocolVersionHybi00
	if req.Method != "GET" {
		return http.StatusMethodNotAllowed, ErrBadRequestMethod
	}
	// HTTP version can be safely ignored.

	if strings.ToLower(req.Header.Get("Upgrade")) != "websocket" ||
		strings.ToLower(req.Header.Get("Connection")) != "upgrade" {
		return http.StatusBadRequest, ErrNotWebSocket
	}

	// TODO(ukai): check Host
	c.Origin, err = url.ParseRequest(req.Header.Get("Origin"))
	if err != nil {
		return http.StatusBadRequest, err
	}

	key1 := req.Header.Get("Sec-Websocket-Key1")
	if key1 == "" {
		return http.StatusBadRequest, ErrChallengeResponse
	}
	key2 := req.Header.Get("Sec-Websocket-Key2")
	if key2 == "" {
		return http.StatusBadRequest, ErrChallengeResponse
	}
	key3 := make([]byte, 8)
	if _, err := io.ReadFull(buf, key3); err != nil {
		return http.StatusBadRequest, ErrChallengeResponse
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

	// Step 4. get key number in Sec-WebSocket-Key<n> fields.
	keyNumber1 := getKeyNumber(key1)
	keyNumber2 := getKeyNumber(key2)

	// Step 5. get number of spaces in Sec-WebSocket-Key<n> fields.
	space1 := uint32(strings.Count(key1, " "))
	space2 := uint32(strings.Count(key2, " "))
	if space1 == 0 || space2 == 0 {
		return http.StatusBadRequest, ErrChallengeResponse
	}

	// Step 6. key number must be an integral multiple of spaces.
	if keyNumber1%space1 != 0 || keyNumber2%space2 != 0 {
		return http.StatusBadRequest, ErrChallengeResponse
	}

	// Step 7. let part be key number divided by spaces.
	part1 := keyNumber1 / space1
	part2 := keyNumber2 / space2

	// Step 8. let challenge be concatenation of part1, part2 and key3.
	// Step 9. get MD5 fingerprint of challenge.
	c.challengeResponse, err = getChallengeResponse(part1, part2, key3)
	if err != nil {
		return http.StatusInternalServerError, err
	}
	protocol := strings.TrimSpace(req.Header.Get("Sec-Websocket-Protocol"))
	protocols := strings.Split(protocol, ",")
	for i := 0; i < len(protocols); i++ {
		c.Protocol = append(c.Protocol, strings.TrimSpace(protocols[i]))
	}

	return http.StatusSwitchingProtocols, nil
}

func (c *hixie76ServerHandshaker) AcceptHandshake(buf *bufio.Writer) (err error) {
	if len(c.Protocol) > 0 {
		if len(c.Protocol) != 1 {
			return ErrBadWebSocketProtocol
		}
	}

	// Step 10. send response status line.
	buf.WriteString("HTTP/1.1 101 WebSocket Protocol Handshake\r\n")
	// Step 11. send response headers.
	buf.WriteString("Upgrade: WebSocket\r\n")
	buf.WriteString("Connection: Upgrade\r\n")
	buf.WriteString("Sec-WebSocket-Origin: " + c.Origin.String() + "\r\n")
	buf.WriteString("Sec-WebSocket-Location: " + c.Location.String() + "\r\n")
	if len(c.Protocol) > 0 {
		buf.WriteString("Sec-WebSocket-Protocol: " + c.Protocol[0] + "\r\n")
	}
	// Step 12. send CRLF.
	buf.WriteString("\r\n")
	// Step 13. send response data.
	buf.Write(c.challengeResponse)
	return buf.Flush()
}

func (c *hixie76ServerHandshaker) NewServerConn(buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) (conn *Conn) {
	return newHixieServerConn(c.Config, buf, rwc, request)
}

// A hixie75ServerHandshaker performs a server handshake using
// hixie draft 75 protocol.
type hixie75ServerHandshaker struct {
	*Config
}

func (c *hixie75ServerHandshaker) ReadHandshake(buf *bufio.Reader, req *http.Request) (code int, err error) {
	c.Version = ProtocolVersionHixie75
	if req.Method != "GET" || req.Proto != "HTTP/1.1" {
		return http.StatusMethodNotAllowed, ErrBadRequestMethod
	}
	if req.Header.Get("Upgrade") != "WebSocket" {
		return http.StatusBadRequest, ErrNotWebSocket
	}
	if req.Header.Get("Connection") != "Upgrade" {
		return http.StatusBadRequest, ErrNotWebSocket
	}
	c.Origin, err = url.ParseRequest(strings.TrimSpace(req.Header.Get("Origin")))
	if err != nil {
		return http.StatusBadRequest, err
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
	protocol := strings.TrimSpace(req.Header.Get("Websocket-Protocol"))
	protocols := strings.Split(protocol, ",")
	for i := 0; i < len(protocols); i++ {
		c.Protocol = append(c.Protocol, strings.TrimSpace(protocols[i]))
	}

	return http.StatusSwitchingProtocols, nil
}

func (c *hixie75ServerHandshaker) AcceptHandshake(buf *bufio.Writer) (err error) {
	if len(c.Protocol) > 0 {
		if len(c.Protocol) != 1 {
			return ErrBadWebSocketProtocol
		}
	}

	buf.WriteString("HTTP/1.1 101 Web Socket Protocol Handshake\r\n")
	buf.WriteString("Upgrade: WebSocket\r\n")
	buf.WriteString("Connection: Upgrade\r\n")
	buf.WriteString("WebSocket-Origin: " + c.Origin.String() + "\r\n")
	buf.WriteString("WebSocket-Location: " + c.Location.String() + "\r\n")
	if len(c.Protocol) > 0 {
		buf.WriteString("WebSocket-Protocol: " + c.Protocol[0] + "\r\n")
	}
	buf.WriteString("\r\n")
	return buf.Flush()
}

func (c *hixie75ServerHandshaker) NewServerConn(buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) (conn *Conn) {
	return newHixieServerConn(c.Config, buf, rwc, request)
}

// newHixieServerConn returns a new WebSocket connection speaking hixie draft protocol.
func newHixieServerConn(config *Config, buf *bufio.ReadWriter, rwc io.ReadWriteCloser, request *http.Request) *Conn {
	return newHixieConn(config, buf, rwc, request)
}
