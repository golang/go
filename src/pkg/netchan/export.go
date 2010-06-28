// Copyright 2010 The Go Authors. All rights reserved.
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
	as if they are buffered channels of at least one element.
*/
package netchan

// BUG: can't use range clause to receive when using ImportNValues with N non-zero.

import (
	"log"
	"net"
	"os"
	"reflect"
	"sync"
)

// Export

// A channel and its associated information: a direction plus
// a handy marshaling place for its data.
type exportChan struct {
	ch  *reflect.ChanValue
	dir Dir
	ptr *reflect.PtrValue // a pointer value we can point at each new received item
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

// Wait for incoming connections, start a new runner for each
func (exp *Exporter) listen() {
	for {
		conn, err := exp.listener.Accept()
		if err != nil {
			log.Stderr("exporter.listen:", err)
			break
		}
		client := newClient(exp, conn)
		go client.run()
	}
}

func (client *expClient) sendError(hdr *header, err string) {
	error := &error{err}
	log.Stderr("export:", error.error)
	client.encode(hdr, payError, error) // ignore any encode error, hope client gets it
}

func (client *expClient) getChan(hdr *header, dir Dir) *exportChan {
	exp := client.exp
	exp.chanLock.Lock()
	ech, ok := exp.chans[hdr.name]
	exp.chanLock.Unlock()
	if !ok {
		client.sendError(hdr, "no such channel: "+hdr.name)
		return nil
	}
	if ech.dir != dir {
		client.sendError(hdr, "wrong direction for channel: "+hdr.name)
		return nil
	}
	return ech
}

// Manage sends and receives for a single client.  For each (client Recv) request,
// this will launch a serveRecv goroutine to deliver the data for that channel,
// while (client Send) requests are handled as data arrives from the client.
func (client *expClient) run() {
	hdr := new(header)
	req := new(request)
	error := new(error)
	for {
		if err := client.decode(hdr); err != nil {
			log.Stderr("error decoding client header:", err)
			// TODO: tear down connection
			return
		}
		switch hdr.payloadType {
		case payRequest:
			if err := client.decode(req); err != nil {
				log.Stderr("error decoding client request:", err)
				// TODO: tear down connection
				return
			}
			switch req.dir {
			case Recv:
				go client.serveRecv(*hdr, req.count)
			case Send:
				// Request to send is clear as a matter of protocol
				// but not actually used by the implementation.
				// The actual sends will have payload type payData.
				// TODO: manage the count?
			default:
				error.error = "export request: can't handle channel direction"
				log.Stderr(error.error, req.dir)
				client.encode(hdr, payError, error)
			}
		case payData:
			client.serveSend(*hdr)
		}
	}
}

// Send all the data on a single channel to a client asking for a Recv.
// The header is passed by value to avoid issues of overwriting.
func (client *expClient) serveRecv(hdr header, count int) {
	ech := client.getChan(&hdr, Send)
	if ech == nil {
		return
	}
	for {
		val := ech.ch.Recv()
		if ech.ch.Closed() {
			client.sendError(&hdr, os.EOF.String())
			break
		}
		if err := client.encode(&hdr, payData, val.Interface()); err != nil {
			log.Stderr("error encoding client response:", err)
			client.sendError(&hdr, err.String())
			break
		}
		if count > 0 {
			if count--; count == 0 {
				break
			}
		}
	}
}

// Receive and deliver locally one item from a client asking for a Send
// The header is passed by value to avoid issues of overwriting.
func (client *expClient) serveSend(hdr header) {
	ech := client.getChan(&hdr, Recv)
	if ech == nil {
		return
	}
	// Create a new value for each received item.
	val := reflect.MakeZero(ech.ptr.Type().(*reflect.PtrType).Elem())
	ech.ptr.PointTo(val)
	if err := client.decode(ech.ptr.Interface()); err != nil {
		log.Stderr("exporter value decode:", err)
		return
	}
	ech.ch.Send(val)
	// TODO count
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
// TODO: fix reflection so we can eliminate the need for pT.
func (exp *Exporter) Export(name string, chT interface{}, dir Dir, pT interface{}) os.Error {
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
	ptr := reflect.MakeZero(reflect.Typeof(pT)).(*reflect.PtrValue)
	exp.chans[name] = &exportChan{ch, dir, ptr}
	return nil
}
