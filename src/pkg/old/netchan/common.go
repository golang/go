// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"errors"
	"gob"
	"io"
	"reflect"
	"sync"
	"time"
)

// The direction of a connection from the client's perspective.
type Dir int

const (
	Recv Dir = iota
	Send
)

func (dir Dir) String() string {
	switch dir {
	case Recv:
		return "Recv"
	case Send:
		return "Send"
	}
	return "???"
}

// Payload types
const (
	payRequest = iota // request structure follows
	payError          // error structure follows
	payData           // user payload follows
	payAck            // acknowledgement; no payload
	payClosed         // channel is now closed
	payAckSend        // payload has been delivered.
)

// A header is sent as a prefix to every transmission.  It will be followed by
// a request structure, an error structure, or an arbitrary user payload structure.
type header struct {
	Id          int
	PayloadType int
	SeqNum      int64
}

// Sent with a header once per channel from importer to exporter to report
// that it wants to bind to a channel with the specified direction for count
// messages, with space for size buffered values. If count is -1, it means unlimited.
type request struct {
	Name  string
	Count int64
	Size  int
	Dir   Dir
}

// Sent with a header to report an error.
type error_ struct {
	Error string
}

// Used to unify management of acknowledgements for import and export.
type unackedCounter interface {
	unackedCount() int64
	ack() int64
	seq() int64
}

// A channel and its direction.
type chanDir struct {
	ch  reflect.Value
	dir Dir
}

// clientSet contains the objects and methods needed for tracking
// clients of an exporter and draining outstanding messages.
type clientSet struct {
	mu      sync.Mutex // protects access to channel and client maps
	names   map[string]*chanDir
	clients map[unackedCounter]bool
}

// Mutex-protected encoder and decoder pair.
type encDec struct {
	decLock sync.Mutex
	dec     *gob.Decoder
	encLock sync.Mutex
	enc     *gob.Encoder
}

func newEncDec(conn io.ReadWriter) *encDec {
	return &encDec{
		dec: gob.NewDecoder(conn),
		enc: gob.NewEncoder(conn),
	}
}

// Decode an item from the connection.
func (ed *encDec) decode(value reflect.Value) error {
	ed.decLock.Lock()
	err := ed.dec.DecodeValue(value)
	if err != nil {
		// TODO: tear down connection?
	}
	ed.decLock.Unlock()
	return err
}

// Encode a header and payload onto the connection.
func (ed *encDec) encode(hdr *header, payloadType int, payload interface{}) error {
	ed.encLock.Lock()
	hdr.PayloadType = payloadType
	err := ed.enc.Encode(hdr)
	if err == nil {
		if payload != nil {
			err = ed.enc.Encode(payload)
		}
	}
	if err != nil {
		// TODO: tear down connection if there is an error?
	}
	ed.encLock.Unlock()
	return err
}

// See the comment for Exporter.Drain.
func (cs *clientSet) drain(timeout int64) error {
	startTime := time.Nanoseconds()
	for {
		pending := false
		cs.mu.Lock()
		// Any messages waiting for a client?
		for _, chDir := range cs.names {
			if chDir.ch.Len() > 0 {
				pending = true
			}
		}
		// Any unacknowledged messages?
		for client := range cs.clients {
			n := client.unackedCount()
			if n > 0 { // Check for > rather than != just to be safe.
				pending = true
				break
			}
		}
		cs.mu.Unlock()
		if !pending {
			break
		}
		if timeout > 0 && time.Nanoseconds()-startTime >= timeout {
			return errors.New("timeout")
		}
		time.Sleep(100 * 1e6) // 100 milliseconds
	}
	return nil
}

// See the comment for Exporter.Sync.
func (cs *clientSet) sync(timeout int64) error {
	startTime := time.Nanoseconds()
	// seq remembers the clients and their seqNum at point of entry.
	seq := make(map[unackedCounter]int64)
	for client := range cs.clients {
		seq[client] = client.seq()
	}
	for {
		pending := false
		cs.mu.Lock()
		// Any unacknowledged messages?  Look only at clients that existed
		// when we started and are still in this client set.
		for client := range seq {
			if _, ok := cs.clients[client]; ok {
				if client.ack() < seq[client] {
					pending = true
					break
				}
			}
		}
		cs.mu.Unlock()
		if !pending {
			break
		}
		if timeout > 0 && time.Nanoseconds()-startTime >= timeout {
			return errors.New("timeout")
		}
		time.Sleep(100 * 1e6) // 100 milliseconds
	}
	return nil
}

// A netChan represents a channel imported or exported
// on a single connection. Flow is controlled by the receiving
// side by sending payAckSend messages when values
// are delivered into the local channel.
type netChan struct {
	*chanDir
	name   string
	id     int
	size   int // buffer size of channel.
	closed bool

	// sender-specific state
	ackCh chan bool // buffered with space for all the acks we need
	space int       // available space.

	// receiver-specific state
	sendCh chan reflect.Value // buffered channel of values received from other end.
	ed     *encDec            // so that we can send acks.
	count  int64              // number of values still to receive.
}

// Create a new netChan with the given name (only used for
// messages), id, direction, buffer size, and count.
// The connection to the other side is represented by ed.
func newNetChan(name string, id int, ch *chanDir, ed *encDec, size int, count int64) *netChan {
	c := &netChan{chanDir: ch, name: name, id: id, size: size, ed: ed, count: count}
	if c.dir == Send {
		c.ackCh = make(chan bool, size)
		c.space = size
	}
	return c
}

// Close the channel.
func (nch *netChan) close() {
	if nch.closed {
		return
	}
	if nch.dir == Recv {
		if nch.sendCh != nil {
			// If the sender goroutine is active, close the channel to it.
			// It will close nch.ch when it can.
			close(nch.sendCh)
		} else {
			nch.ch.Close()
		}
	} else {
		nch.ch.Close()
		close(nch.ackCh)
	}
	nch.closed = true
}

// Send message from remote side to local receiver.
func (nch *netChan) send(val reflect.Value) {
	if nch.dir != Recv {
		panic("send on wrong direction of channel")
	}
	if nch.sendCh == nil {
		// If possible, do local send directly and ack immediately.
		if nch.ch.TrySend(val) {
			nch.sendAck()
			return
		}
		// Start sender goroutine to manage delayed delivery of values.
		nch.sendCh = make(chan reflect.Value, nch.size)
		go nch.sender()
	}
	select {
	case nch.sendCh <- val:
		// ok
	default:
		// TODO: should this be more resilient?
		panic("netchan: remote sender sent more values than allowed")
	}
}

// sendAck sends an acknowledgment that a message has left
// the channel's buffer. If the messages remaining to be sent
// will fit in the channel's buffer, then we don't
// need to send an ack.
func (nch *netChan) sendAck() {
	if nch.count < 0 || nch.count > int64(nch.size) {
		nch.ed.encode(&header{Id: nch.id}, payAckSend, nil)
	}
	if nch.count > 0 {
		nch.count--
	}
}

// The sender process forwards items from the sending queue
// to the destination channel, acknowledging each item.
func (nch *netChan) sender() {
	if nch.dir != Recv {
		panic("sender on wrong direction of channel")
	}
	// When Exporter.Hangup is called, the underlying channel is closed,
	// and so we may get a "too many operations on closed channel" error
	// if there are outstanding messages in sendCh.
	// Make sure that this doesn't panic the whole program.
	defer func() {
		if r := recover(); r != nil {
			// TODO check that r is "too many operations", otherwise re-panic.
		}
	}()
	for v := range nch.sendCh {
		nch.ch.Send(v)
		nch.sendAck()
	}
	nch.ch.Close()
}

// Receive value from local side for sending to remote side.
func (nch *netChan) recv() (val reflect.Value, ok bool) {
	if nch.dir != Send {
		panic("recv on wrong direction of channel")
	}

	if nch.space == 0 {
		// Wait for buffer space.
		<-nch.ackCh
		nch.space++
	}
	nch.space--
	return nch.ch.Recv()
}

// acked is called when the remote side indicates that
// a value has been delivered.
func (nch *netChan) acked() {
	if nch.dir != Send {
		panic("recv on wrong direction of channel")
	}
	select {
	case nch.ackCh <- true:
		// ok
	default:
		// TODO: should this be more resilient?
		panic("netchan: remote receiver sent too many acks")
	}
}
