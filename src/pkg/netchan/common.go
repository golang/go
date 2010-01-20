// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"gob"
	"log"
	"net"
	"os"
	"sync"
)

type Dir int

const (
	Recv Dir = iota
	Send
)

// Mutex-protected encoder and decoder pair

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

func (ed *encDec) decode(e interface{}) os.Error {
	ed.decLock.Lock()
	defer ed.decLock.Unlock()
	err := ed.dec.Decode(e)
	if err != nil {
		log.Stderr("exporter decode:", err)
		// TODO: tear down connection
		return err
	}
	return nil
}

func (ed *encDec) encode(e0, e1 interface{}) os.Error {
	ed.encLock.Lock()
	defer ed.encLock.Unlock()
	err := ed.enc.Encode(e0)
	if err == nil && e1 != nil {
		err = ed.enc.Encode(e1)
	}
	if err != nil {
		log.Stderr("exporter encode:", err)
		// TODO: tear down connection?
		return err
	}
	return nil
}
