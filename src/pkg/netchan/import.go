// Copyright 2010 The Go Authors. All rights reserved.
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

// A channel and its associated information: a template value and direction,
// plus a handy marshaling place for its data.
type importChan struct {
	ch  *reflect.ChanValue
	dir Dir
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

// shutdown closes all channels for which we are receiving data from the remote side.
func (imp *Importer) shutdown() {
	imp.chanLock.Lock()
	for _, ich := range imp.chans {
		if ich.dir == Recv {
			ich.ch.Close()
		}
	}
	imp.chanLock.Unlock()
}

// Handle the data from a single imported data stream, which will
// have the form
//	(response, data)*
// The response identifies by name which channel is transmitting data.
func (imp *Importer) run() {
	// Loop on responses; requests are sent by ImportNValues()
	hdr := new(header)
	hdrValue := reflect.NewValue(hdr)
	err := new(error)
	errValue := reflect.NewValue(err)
	for {
		if e := imp.decode(hdrValue); e != nil {
			log.Stderr("importer header:", e)
			imp.shutdown()
			return
		}
		switch hdr.payloadType {
		case payData:
			// done lower in loop
		case payError:
			if e := imp.decode(errValue); e != nil {
				log.Stderr("importer error:", e)
				return
			}
			if err.error != "" {
				log.Stderr("importer response error:", err.error)
				imp.shutdown()
				return
			}
		default:
			log.Stderr("unexpected payload type:", hdr.payloadType)
			return
		}
		imp.chanLock.Lock()
		ich, ok := imp.chans[hdr.name]
		imp.chanLock.Unlock()
		if !ok {
			log.Stderr("unknown name in request:", hdr.name)
			return
		}
		if ich.dir != Recv {
			log.Stderr("cannot happen: receive from non-Recv channel")
			return
		}
		// Create a new value for each received item.
		value := reflect.MakeZero(ich.ch.Type().(*reflect.ChanType).Elem())
		if e := imp.decode(value); e != nil {
			log.Stderr("importer value decode:", e)
			return
		}
		ich.ch.Send(value)
	}
}

// Import imports a channel of the given type and specified direction.
// It is equivalent to ImportNValues with a count of -1, meaning unbounded.
func (imp *Importer) Import(name string, chT interface{}, dir Dir) os.Error {
	return imp.ImportNValues(name, chT, dir, 0)
}

// ImportNValues imports a channel of the given type and specified direction
// and then receives or transmits up to n values on that channel.  A value of
// n==-1 implies an unbounded number of values.  The channel to be bound to
// the remote site's channel is provided in the call and may be of arbitrary
// channel type.
// Despite the literal signature, the effective signature is
//	ImportNValues(name string, chT chan T, dir Dir, n int) os.Error
// Example usage:
//	imp, err := NewImporter("tcp", "netchanserver.mydomain.com:1234")
//	if err != nil { log.Exit(err) }
//	ch := make(chan myType)
//	err := imp.ImportNValues("name", ch, Recv, 1)
//	if err != nil { log.Exit(err) }
//	fmt.Printf("%+v\n", <-ch)
func (imp *Importer) ImportNValues(name string, chT interface{}, dir Dir, n int) os.Error {
	ch, err := checkChan(chT, dir)
	if err != nil {
		return err
	}
	imp.chanLock.Lock()
	defer imp.chanLock.Unlock()
	_, present := imp.chans[name]
	if present {
		return os.ErrorString("channel name already being imported:" + name)
	}
	imp.chans[name] = &importChan{ch, dir}
	// Tell the other side about this channel.
	hdr := new(header)
	hdr.name = name
	hdr.payloadType = payRequest
	req := new(request)
	req.dir = dir
	req.count = n
	if err := imp.encode(hdr, payRequest, req); err != nil {
		log.Stderr("importer request encode:", err)
		return err
	}
	if dir == Send {
		go func() {
			for i := 0; n == 0 || i < n; i++ {
				val := ch.Recv()
				if err := imp.encode(hdr, payData, val.Interface()); err != nil {
					log.Stderr("error encoding client response:", err)
					return
				}
			}
		}()
	}
	return nil
}
