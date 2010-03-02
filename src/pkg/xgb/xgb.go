// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The XGB package implements the X11 core protocol.
// It is based on XCB: http://xcb.freedesktop.org/
package xgb

import (
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
)

// A Conn represents a connection to an X server.
// Only one goroutine should use a Conn's methods at a time.
type Conn struct {
	host          string
	conn          net.Conn
	nextId        Id
	nextCookie    Cookie
	replies       map[Cookie][]byte
	events        queue
	err           os.Error
	display       string
	defaultScreen int
	scratch       [32]byte
	Setup         SetupInfo
}

// Id is used for all X identifiers, such as windows, pixmaps, and GCs.
type Id uint32

// Cookies are the sequence numbers used to pair replies up with their requests
type Cookie uint16

type Keysym uint32
type Timestamp uint32

// Event is an interface that can contain any of the events returned by the server.
// Use a type assertion switch to extract the Event structs.
type Event interface{}

// Error contains protocol errors returned to us by the X server.
type Error struct {
	Detail uint8
	Major  uint8
	Minor  uint16
	Cookie Cookie
	Id     Id
}

func (e *Error) String() string {
	return fmt.Sprintf("Bad%s (major=%d minor=%d cookie=%d id=0x%x)",
		errorNames[e.Detail], e.Major, e.Minor, e.Cookie, e.Id)
}

// NewID generates a new unused ID for use with requests like CreateWindow.
func (c *Conn) NewId() Id {
	id := c.nextId
	// TODO: handle ID overflow
	c.nextId++
	return id
}

// Pad a length to align on 4 bytes.
func pad(n int) int { return (n + 3) & ^3 }

func put16(buf []byte, v uint16) {
	buf[0] = byte(v)
	buf[1] = byte(v >> 8)
}

func put32(buf []byte, v uint32) {
	buf[0] = byte(v)
	buf[1] = byte(v >> 8)
	buf[2] = byte(v >> 16)
	buf[3] = byte(v >> 24)
}

func get16(buf []byte) uint16 {
	v := uint16(buf[0])
	v |= uint16(buf[1]) << 8
	return v
}

func get32(buf []byte) uint32 {
	v := uint32(buf[0])
	v |= uint32(buf[1]) << 8
	v |= uint32(buf[2]) << 16
	v |= uint32(buf[3]) << 24
	return v
}

// Voodoo to count the number of bits set in a value list mask.
func popCount(mask0 int) int {
	mask := uint32(mask0)
	n := 0
	for i := uint32(0); i < 32; i++ {
		if mask&(1<<i) != 0 {
			n++
		}
	}
	return n
}

// A simple queue used to stow away events.
type queue struct {
	data [][]byte
	a, b int
}

func (q *queue) queue(item []byte) {
	if q.b == len(q.data) {
		if q.a > 0 {
			copy(q.data, q.data[q.a:q.b])
			q.a, q.b = 0, q.b-q.a
		} else {
			newData := make([][]byte, (len(q.data)*3)/2)
			copy(newData, q.data)
			q.data = newData
		}
	}
	q.data[q.b] = item
	q.b++
}

func (q *queue) dequeue() []byte {
	if q.a < q.b {
		item := q.data[q.a]
		q.a++
		return item
	}
	return nil
}

// sendRequest sends a request to the server and return its associated sequence number, or cookie.
// It is only used to send the fixed length portion of the request, sendBytes and friends are used
// to send any additional variable length data.
func (c *Conn) sendRequest(buf []byte) Cookie {
	if _, err := c.conn.Write(buf); err != nil {
		fmt.Fprintf(os.Stderr, "x protocol write error: %s\n", err)
		c.err = err
	}
	cookie := c.nextCookie
	c.nextCookie++
	return cookie
}

// sendPadding sends enough bytes to align to a 4-byte border.
// It is used to pad the variable length data that is used with some requests.
func (c *Conn) sendPadding(n int) {
	x := pad(n) - n
	if x > 0 {
		_, err := c.conn.Write(c.scratch[0:x])
		if err != nil {
			fmt.Fprintf(os.Stderr, "x protocol write error: %s\n", err)
			c.err = err
		}
	}
}

// sendBytes sends a byte slice as variable length data after the fixed portion of a request,
// along with any necessary padding.
func (c *Conn) sendBytes(buf []byte) {
	if _, err := c.conn.Write(buf); err != nil {
		fmt.Fprintf(os.Stderr, "x protocol write error: %s\n", err)
		c.err = err
	}
	c.sendPadding(len(buf))
}

func (c *Conn) sendString(str string) { c.sendBytes([]byte(str)) }

// sendUInt32s sends a list of 32-bit integers as variable length data.
func (c *Conn) sendUInt32List(list []uint32) {
	buf := make([]byte, len(list)*4)
	for i := 0; i < len(list); i++ {
		put32(buf[i*4:], list[i])
	}
	c.sendBytes(buf)
}

func (c *Conn) sendIdList(list []Id, length int) {
	buf := make([]byte, length*4)
	for i := 0; i < length; i++ {
		put32(buf[i*4:], uint32(list[i]))
	}
	c.sendBytes(buf)
}

func (c *Conn) sendKeysymList(list []Keysym, length int) {
	buf := make([]byte, length*4)
	for i := 0; i < length; i++ {
		put32(buf[i*4:], uint32(list[i]))
	}
	c.sendBytes(buf)
}

// readNextReply reads and processes the next server reply.
// If it is a protocol error then it is returned as an Error.
// Events are pushed onto the event queue and replies to requests
// are stashed away in a map indexed by the sequence number.
func (c *Conn) readNextReply() os.Error {
	buf := make([]byte, 32)
	if _, err := io.ReadFull(c.conn, buf); err != nil {
		fmt.Fprintf(os.Stderr, "x protocol read error: %s\n", err)
		return err
	}

	switch buf[0] {
	case 0:
		err := &Error{
			Detail: buf[1],
			Cookie: Cookie(get16(buf[2:])),
			Id:     Id(get32(buf[4:])),
			Minor:  get16(buf[8:]),
			Major:  buf[10],
		}
		fmt.Fprintf(os.Stderr, "x protocol error: %s\n", err)
		return err

	case 1:
		seq := Cookie(get16(buf[2:]))
		size := get32(buf[4:])
		if size > 0 {
			bigbuf := make([]byte, 32+size*4, 32+size*4)
			copy(bigbuf[0:32], buf)
			if _, err := io.ReadFull(c.conn, bigbuf[32:]); err != nil {
				fmt.Fprintf(os.Stderr, "x protocol read error: %s\n", err)
				return err
			}
			c.replies[seq] = bigbuf
		} else {
			c.replies[seq] = buf
		}

	default:
		c.events.queue(buf)
	}

	return nil
}

// waitForReply looks for a reply in the map indexed by sequence number.
// If the reply is not in the map it will block while reading replies from the server
// until the reply is found or an error occurs.
func (c *Conn) waitForReply(cookie Cookie) ([]byte, os.Error) {
	for {
		if reply, ok := c.replies[cookie]; ok {
			c.replies[cookie] = reply, false
			return reply, nil
		}
		if err := c.readNextReply(); err != nil {
			return nil, err
		}
	}
	panic("unreachable")
}

// WaitForEvent returns the next event from the server.
// It will block until an event is available.
func (c *Conn) WaitForEvent() (Event, os.Error) {
	for {
		if reply := c.events.dequeue(); reply != nil {
			return parseEvent(reply)
		}
		if err := c.readNextReply(); err != nil {
			return nil, err
		}
	}
	panic("unreachable")
}

// PollForEvent returns the next event from the server if one is available in the internal queue.
// It will not read from the connection, so you must call WaitForEvent to receive new events.
// Only use this function to empty the queue without blocking.
func (c *Conn) PollForEvent() (Event, os.Error) {
	if reply := c.events.dequeue(); reply != nil {
		return parseEvent(reply)
	}
	return nil, nil
}

// Dial connects to the X server given in the 'display' string.
// If 'display' is empty it will be taken from os.Getenv("DISPLAY").
//
// Examples:
//	Dial(":1")                 // connect to net.Dial("unix", "", "/tmp/.X11-unix/X1")
//	Dial("/tmp/launch-123/:0") // connect to net.Dial("unix", "", "/tmp/launch-123/:0")
//	Dial("hostname:2.1")       // connect to net.Dial("tcp", "", "hostname:6002")
//	Dial("tcp/hostname:1.0")   // connect to net.Dial("tcp", "", "hostname:6001")
func Dial(display string) (*Conn, os.Error) {
	c, err := connect(display)
	if err != nil {
		return nil, err
	}

	// Get authentication data
	authName, authData, err := readAuthority(c.host, c.display)
	if err != nil {
		return nil, err
	}

	// Assume that the authentication protocol is "MIT-MAGIC-COOKIE-1".
	if authName != "MIT-MAGIC-COOKIE-1" || len(authData) != 16 {
		return nil, os.NewError("unsupported auth protocol " + authName)
	}

	buf := make([]byte, 12+pad(len(authName))+pad(len(authData)))
	buf[0] = 0x6c
	buf[1] = 0
	put16(buf[2:], 11)
	put16(buf[4:], 0)
	put16(buf[6:], uint16(len(authName)))
	put16(buf[8:], uint16(len(authData)))
	put16(buf[10:], 0)
	copy(buf[12:], []byte(authName))
	copy(buf[12+pad(len(authName)):], authData)
	if _, err = c.conn.Write(buf); err != nil {
		return nil, err
	}

	head := make([]byte, 8)
	if _, err = io.ReadFull(c.conn, head[0:8]); err != nil {
		return nil, err
	}
	code := head[0]
	reasonLen := head[1]
	major := get16(head[2:])
	minor := get16(head[4:])
	dataLen := get16(head[6:])

	if major != 11 || minor != 0 {
		return nil, os.NewError(fmt.Sprintf("x protocol version mismatch: %d.%d", major, minor))
	}

	buf = make([]byte, int(dataLen)*4+8, int(dataLen)*4+8)
	copy(buf, head)
	if _, err = io.ReadFull(c.conn, buf[8:]); err != nil {
		return nil, err
	}

	if code == 0 {
		reason := buf[8 : 8+reasonLen]
		return nil, os.NewError(fmt.Sprintf("x protocol authentication refused: %s", string(reason)))
	}

	getSetupInfo(buf, &c.Setup)

	if c.defaultScreen >= len(c.Setup.Roots) {
		c.defaultScreen = 0
	}

	c.nextId = Id(c.Setup.ResourceIdBase)
	c.nextCookie = 1
	c.replies = make(map[Cookie][]byte)
	c.events = queue{make([][]byte, 100), 0, 0}
	return c, nil
}

// Close closes the connection to the X server.
func (c *Conn) Close() { c.conn.Close() }

// DefaultScreen returns the Screen info for the default screen, which is
// 0 or the one given in the display argument to Dial.
func (c *Conn) DefaultScreen() *ScreenInfo { return &c.Setup.Roots[c.defaultScreen] }


// ClientMessageData holds the data from a client message,
// duplicated in three forms because Go doesn't have unions.
type ClientMessageData struct {
	Data8  [20]byte
	Data16 [10]uint16
	Data32 [5]uint32
}

func getClientMessageData(b []byte, v *ClientMessageData) int {
	copy(&v.Data8, b)
	for i := 0; i < 10; i++ {
		v.Data16[i] = get16(b[i*2:])
	}
	for i := 0; i < 5; i++ {
		v.Data32[i] = get32(b[i*4:])
	}
	return 20
}

func connect(display string) (*Conn, os.Error) {
	if len(display) == 0 {
		display = os.Getenv("DISPLAY")
	}

	display0 := display
	if len(display) == 0 {
		return nil, os.NewError("empty display string")
	}

	colonIdx := strings.LastIndex(display, ":")
	if colonIdx < 0 {
		return nil, os.NewError("bad display string: " + display0)
	}

	var protocol, socket string
	c := new(Conn)

	if display[0] == '/' {
		socket = display[0:colonIdx]
	} else {
		slashIdx := strings.LastIndex(display, "/")
		if slashIdx >= 0 {
			protocol = display[0:slashIdx]
			c.host = display[slashIdx+1 : colonIdx]
		} else {
			c.host = display[0:colonIdx]
		}
	}

	display = display[colonIdx+1 : len(display)]
	if len(display) == 0 {
		return nil, os.NewError("bad display string: " + display0)
	}

	var scr string
	dotIdx := strings.LastIndex(display, ".")
	if dotIdx < 0 {
		c.display = display[0:]
	} else {
		c.display = display[0:dotIdx]
		scr = display[dotIdx+1:]
	}

	dispnum, err := strconv.Atoui(c.display)
	if err != nil {
		return nil, os.NewError("bad display string: " + display0)
	}

	if len(scr) != 0 {
		c.defaultScreen, err = strconv.Atoi(scr)
		if err != nil {
			return nil, os.NewError("bad display string: " + display0)
		}
	}

	// Connect to server
	if len(socket) != 0 {
		c.conn, err = net.Dial("unix", "", socket+":"+c.display)
	} else if len(c.host) != 0 {
		if protocol == "" {
			protocol = "tcp"
		}
		c.conn, err = net.Dial(protocol, "", c.host+":"+strconv.Uitoa(6000+dispnum))
	} else {
		c.conn, err = net.Dial("unix", "", "/tmp/.X11-unix/X"+c.display)
	}

	if err != nil {
		return nil, os.NewError("cannot connect to " + display0 + ": " + err.String())
	}
	return c, nil
}
