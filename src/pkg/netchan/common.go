// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"gob"
	"net"
	"os"
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
)

// A header is sent as a prefix to every transmission.  It will be followed by
// a request structure, an error structure, or an arbitrary user payload structure.
type header struct {
	name        string
	payloadType int
	seqNum      int64
}

// Sent with a header once per channel from importer to exporter to report
// that it wants to bind to a channel with the specified direction for count
// messages.  If count is -1, it means unlimited.
type request struct {
	count int64
	dir   Dir
}

// Sent with a header to report an error.
type error struct {
	error string
}

// Used to unify management of acknowledgements for import and export.
type unackedCounter interface {
	unackedCount() int64
	ack() int64
	seq() int64
}

// A channel and its direction.
type chanDir struct {
	ch  *reflect.ChanValue
	dir Dir
}

// clientSet contains the objects and methods needed for tracking
// clients of an exporter and draining outstanding messages.
type clientSet struct {
	mu      sync.Mutex // protects access to channel and client maps
	chans   map[string]*chanDir
	clients map[unackedCounter]bool
}

// Mutex-protected encoder and decoder pair.
type encDec struct {
	decLock sync.Mutex
	dec     *gob.Decoder
	encLock sync.Mutex
	enc     *gob.Encoder
}

func newEncDec(conn net.Conn) *encDec {
	return &encDec{
		dec: gob.NewDecoder(conn),
		enc: gob.NewEncoder(conn),
	}
}

// Decode an item from the connection.
func (ed *encDec) decode(value reflect.Value) os.Error {
	ed.decLock.Lock()
	err := ed.dec.DecodeValue(value)
	if err != nil {
		// TODO: tear down connection?
	}
	ed.decLock.Unlock()
	return err
}

// Encode a header and payload onto the connection.
func (ed *encDec) encode(hdr *header, payloadType int, payload interface{}) os.Error {
	ed.encLock.Lock()
	hdr.payloadType = payloadType
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
func (cs *clientSet) drain(timeout int64) os.Error {
	startTime := time.Nanoseconds()
	for {
		pending := false
		cs.mu.Lock()
		// Any messages waiting for a client?
		for _, chDir := range cs.chans {
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
			return os.ErrorString("timeout")
		}
		time.Sleep(100 * 1e6) // 100 milliseconds
	}
	return nil
}

// See the comment for Exporter.Sync.
func (cs *clientSet) sync(timeout int64) os.Error {
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
			return os.ErrorString("timeout")
		}
		time.Sleep(100 * 1e6) // 100 milliseconds
	}
	return nil
}
