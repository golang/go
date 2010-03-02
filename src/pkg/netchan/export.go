// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The netchan package implements type-safe networked channels:
	it allows the two ends of a channel to appear on different
	computers connected by a network.  It does this by transporting
	data sent to a channel on one machine so it can be recovered
	by a receive of a channel of the same type on the other.

	An exporter publishes a set of channels by name.  An importer
	connects to the exporting machine and imports the channels
	by name. After importing the channels, the two machines can
	use the channels in the usual way.

	Networked channels are not synchronized; they always behave
	as if there is a buffer of at least one element between the
	two machines.

	TODO: at the moment, the exporting machine must send and
	the importing machine must receive.  This restriction will
	be lifted soon.
*/
package netchan

import (
	"log"
	"net"
	"os"
	"reflect"
	"sync"
)

// Export

// A channel and its associated information: a direction
type exportChan struct {
	ch  *reflect.ChanValue
	dir Dir
}

// An Exporter allows a set of channels to be published on a single
// network port.  A single machine may have multiple Exporters
// but they must use different ports.
type Exporter struct {
	listener net.Listener
	chanLock sync.Mutex // protects access to channel map
	chans    map[string]*exportChan
}

type expClient struct {
	*encDec
	exp *Exporter
}

func newClient(exp *Exporter, conn net.Conn) *expClient {
	client := new(expClient)
	client.exp = exp
	client.encDec = newEncDec(conn)
	return client

}

// TODO: ASSUMES EXPORT MEANS SEND

// Sent once per channel from importer to exporter to report that it's listening to a channel
type request struct {
	name  string
	dir   Dir
	count int
}

// Reply to request, sent from exporter to importer on each send.
type response struct {
	name  string
	error string
}

// Wait for incoming connections, start a new runner for each
func (exp *Exporter) listen() {
	for {
		conn, err := exp.listener.Accept()
		if err != nil {
			log.Stderr("exporter.listen:", err)
			break
		}
		log.Stderr("accepted call from", conn.RemoteAddr())
		client := newClient(exp, conn)
		go client.run()
	}
}

// Send a single client all its data.  For each request, this will launch
// a serveRecv goroutine to deliver the data for that channel.
func (client *expClient) run() {
	req := new(request)
	for {
		if err := client.decode(req); err != nil {
			log.Stderr("error decoding client request:", err)
			// TODO: tear down connection
			break
		}
		log.Stderrf("export request: %+v", req)
		if req.dir == Recv {
			go client.serveRecv(req)
		} else {
			log.Stderr("export request: can't handle channel direction", req.dir)
			resp := new(response)
			resp.name = req.name
			resp.error = "export request: can't handle channel direction"
			client.encode(resp, nil)
			break
		}
	}
}

// Send all the data on a single channel to a client asking for a Recv
func (client *expClient) serveRecv(req *request) {
	exp := client.exp
	resp := new(response)
	resp.name = req.name
	var ok bool
	exp.chanLock.Lock()
	ech, ok := exp.chans[req.name]
	exp.chanLock.Unlock()
	if !ok {
		resp.error = "no such channel: " + req.name
		log.Stderr("export:", resp.error)
		client.encode(resp, nil) // ignore any encode error, hope client gets it
		return
	}
	for {
		if ech.dir != Send {
			log.Stderr("TODO: recv export unimplemented")
			break
		}
		val := ech.ch.Recv()
		if err := client.encode(resp, val.Interface()); err != nil {
			log.Stderr("error encoding client response:", err)
			break
		}
		if req.count > 0 {
			req.count--
			if req.count == 0 {
				break
			}
		}
	}
}

// NewExporter creates a new Exporter to export channels
// on the network and local address defined as in net.Listen.
func NewExporter(network, localaddr string) (*Exporter, os.Error) {
	listener, err := net.Listen(network, localaddr)
	if err != nil {
		return nil, err
	}
	e := &Exporter{
		listener: listener,
		chans:    make(map[string]*exportChan),
	}
	go e.listen()
	return e, nil
}

// Addr returns the Exporter's local network address.
func (exp *Exporter) Addr() net.Addr { return exp.listener.Addr() }

func checkChan(chT interface{}, dir Dir) (*reflect.ChanValue, os.Error) {
	chanType, ok := reflect.Typeof(chT).(*reflect.ChanType)
	if !ok {
		return nil, os.ErrorString("not a channel")
	}
	if dir != Send && dir != Recv {
		return nil, os.ErrorString("unknown channel direction")
	}
	switch chanType.Dir() {
	case reflect.BothDir:
	case reflect.SendDir:
		if dir != Recv {
			return nil, os.ErrorString("to import/export with Send, must provide <-chan")
		}
	case reflect.RecvDir:
		if dir != Send {
			return nil, os.ErrorString("to import/export with Recv, must provide chan<-")
		}
	}
	return reflect.NewValue(chT).(*reflect.ChanValue), nil
}

// Export exports a channel of a given type and specified direction.  The
// channel to be exported is provided in the call and may be of arbitrary
// channel type.
// Despite the literal signature, the effective signature is
//	Export(name string, chT chan T, dir Dir)
// where T must be a struct, pointer to struct, etc.
func (exp *Exporter) Export(name string, chT interface{}, dir Dir) os.Error {
	ch, err := checkChan(chT, dir)
	if err != nil {
		return err
	}
	exp.chanLock.Lock()
	defer exp.chanLock.Unlock()
	_, present := exp.chans[name]
	if present {
		return os.ErrorString("channel name already being exported:" + name)
	}
	exp.chans[name] = &exportChan{ch, dir}
	return nil
}
