// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"gob"
	"net"
	"os"
	"sync"
)

// The direction of a connection from the client's perspective.
type Dir int

const (
	Recv Dir = iota
	Send
)

// Payload types
const (
	payRequest = iota // request structure follows
	payError          // error structure follows
	payData           // user payload follows
)

// A header is sent as a prefix to every transmission.  It will be followed by
// a request structure, an error structure, or an arbitrary user payload structure.
type header struct {
	name        string
	payloadType int
}

// Sent with a header once per channel from importer to exporter to report
// that it wants to bind to a channel with the specified direction for count
// messages.  If count is zero, it means unlimited.
type request struct {
	count int
	dir   Dir
}

// Sent with a header to report an error.
type error struct {
	error string
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
func (ed *encDec) decode(e interface{}) os.Error {
	ed.decLock.Lock()
	err := ed.dec.Decode(e)
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
		err = ed.enc.Encode(payload)
	} else {
		// TODO: tear down connection if there is an error?
	}
	ed.encLock.Unlock()
	return err
}
