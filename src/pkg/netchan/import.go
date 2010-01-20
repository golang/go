// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"log"
	"net"
	"os"
	"reflect"
	"sync"
)

// Import

// A channel and its associated information: a template value, direction and a count
type importChan struct {
	ch    *reflect.ChanValue
	dir   Dir
	ptr   *reflect.PtrValue // a pointer value we can point at each new item
	count int
}

// An Importer allows a set of channels to be imported from a single
// remote machine/network port.  A machine may have multiple
// importers, even from the same machine/network port.
type Importer struct {
	*encDec
	conn     net.Conn
	chanLock sync.Mutex // protects access to channel map
	chans    map[string]*importChan
}

// TODO: ASSUMES IMPORT MEANS RECEIVE

// NewImporter creates a new Importer object to import channels
// from an Exporter at the network and remote address as defined in net.Dial.
// The Exporter must be available and serving when the Importer is
// created.
func NewImporter(network, remoteaddr string) (*Importer, os.Error) {
	conn, err := net.Dial(network, "", remoteaddr)
	if err != nil {
		return nil, err
	}
	imp := new(Importer)
	imp.encDec = newEncDec(conn)
	imp.conn = conn
	imp.chans = make(map[string]*importChan)
	go imp.run()
	return imp, nil
}

// Handle the data from a single imported data stream, which will
// have the form
//	(response, data)*
// The response identifies by name which channel is receiving data.
// TODO: allow an importer to send.
func (imp *Importer) run() {
	// Loop on responses; requests are sent by ImportNValues()
	resp := new(response)
	for {
		if err := imp.decode(resp); err != nil {
			log.Stderr("importer response decode:", err)
			break
		}
		if resp.error != "" {
			log.Stderr("importer response error:", resp.error)
			// TODO: tear down connection
			break
		}
		imp.chanLock.Lock()
		ich, ok := imp.chans[resp.name]
		imp.chanLock.Unlock()
		if !ok {
			log.Stderr("unknown name in request:", resp.name)
			break
		}
		if ich.dir != Recv {
			log.Stderr("TODO: import send unimplemented")
			break
		}
		// Create a new value for each received item.
		val := reflect.MakeZero(ich.ptr.Type().(*reflect.PtrType).Elem())
		ich.ptr.PointTo(val)
		if err := imp.decode(ich.ptr.Interface()); err != nil {
			log.Stderr("importer value decode:", err)
			return
		}
		ich.ch.Send(val)
	}
}

// Import imports a channel of the given type and specified direction.
// It is equivalent to ImportNValues with a count of 0, meaning unbounded.
func (imp *Importer) Import(name string, chT interface{}, dir Dir, pT interface{}) os.Error {
	return imp.ImportNValues(name, chT, dir, pT, 0)
}

// ImportNValues imports a channel of the given type and specified direction
// and then receives or transmits up to n values on that channel.  A value of
// n==0 implies an unbounded number of values.  The channel to be bound to
// the remote site's channel is provided in the call and may be of arbitrary
// channel type.
// Despite the literal signature, the effective signature is
//	ImportNValues(name string, chT chan T, dir Dir, pT T)
// where T must be a struct, pointer to struct, etc.  pT may be more indirect
// than the value type of the channel (e.g.  chan T, pT *T) but it must be a
// pointer.
// Example usage:
//	imp, err := NewImporter("tcp", "netchanserver.mydomain.com:1234")
//	if err != nil { log.Exit(err) }
//	ch := make(chan myType)
//	err := imp.ImportNValues("name", ch, Recv, new(myType), 1)
//	if err != nil { log.Exit(err) }
//	fmt.Printf("%+v\n", <-ch)
// (TODO: Can we eliminate the need for pT?)
func (imp *Importer) ImportNValues(name string, chT interface{}, dir Dir, pT interface{}, n int) os.Error {
	ch, err := checkChan(chT, dir)
	if err != nil {
		return err
	}
	// Make sure pT is a pointer (to a pointer...) to a struct.
	rt := reflect.Typeof(pT)
	if _, ok := rt.(*reflect.PtrType); !ok {
		return os.ErrorString("not a pointer:" + rt.String())
	}
	if _, ok := reflect.Indirect(reflect.NewValue(pT)).(*reflect.StructValue); !ok {
		return os.ErrorString("not a pointer to a struct:" + rt.String())
	}
	imp.chanLock.Lock()
	defer imp.chanLock.Unlock()
	_, present := imp.chans[name]
	if present {
		return os.ErrorString("channel name already being imported:" + name)
	}
	ptr := reflect.MakeZero(reflect.Typeof(pT)).(*reflect.PtrValue)
	imp.chans[name] = &importChan{ch, dir, ptr, n}
	// Tell the other side about this channel.
	req := new(request)
	req.name = name
	req.dir = dir
	req.count = n
	if err := imp.encode(req, nil); err != nil {
		log.Stderr("importer request encode:", err)
		return err
	}
	return nil
}
