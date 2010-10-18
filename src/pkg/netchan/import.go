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

// impLog is a logging convenience function.  The first argument must be a string.
func impLog(args ...interface{}) {
	args[0] = "netchan import: " + args[0].(string)
	log.Print(args...)
}

// An Importer allows a set of channels to be imported from a single
// remote machine/network port.  A machine may have multiple
// importers, even from the same machine/network port.
type Importer struct {
	*encDec
	conn     net.Conn
	chanLock sync.Mutex // protects access to channel map
	chans    map[string]*chanDir
	errors   chan os.Error
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
	imp.chans = make(map[string]*chanDir)
	imp.errors = make(chan os.Error, 10)
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
	ackHdr := new(header)
	err := new(error)
	errValue := reflect.NewValue(err)
	for {
		*hdr = header{}
		if e := imp.decode(hdrValue); e != nil {
			impLog("header:", e)
			imp.shutdown()
			return
		}
		switch hdr.payloadType {
		case payData:
			// done lower in loop
		case payError:
			if e := imp.decode(errValue); e != nil {
				impLog("error:", e)
				return
			}
			if err.error != "" {
				impLog("response error:", err.error)
				if sent := imp.errors <- os.ErrorString(err.error); !sent {
					imp.shutdown()
					return
				}
				continue // errors are not acknowledged.
			}
		case payClosed:
			ich := imp.getChan(hdr.name)
			if ich != nil {
				ich.ch.Close()
			}
			continue // closes are not acknowledged.
		default:
			impLog("unexpected payload type:", hdr.payloadType)
			return
		}
		ich := imp.getChan(hdr.name)
		if ich == nil {
			continue
		}
		if ich.dir != Recv {
			impLog("cannot happen: receive from non-Recv channel")
			return
		}
		// Acknowledge receipt
		ackHdr.name = hdr.name
		ackHdr.seqNum = hdr.seqNum
		imp.encode(ackHdr, payAck, nil)
		// Create a new value for each received item.
		value := reflect.MakeZero(ich.ch.Type().(*reflect.ChanType).Elem())
		if e := imp.decode(value); e != nil {
			impLog("importer value decode:", e)
			return
		}
		ich.ch.Send(value)
	}
}

func (imp *Importer) getChan(name string) *chanDir {
	imp.chanLock.Lock()
	ich := imp.chans[name]
	imp.chanLock.Unlock()
	if ich == nil {
		impLog("unknown name in netchan request:", name)
		return nil
	}
	return ich
}

// Errors returns a channel from which transmission and protocol errors
// can be read. Clients of the importer are not required to read the error
// channel for correct execution. However, if too many errors occur
// without being read from the error channel, the importer will shut down.
func (imp *Importer) Errors() chan os.Error {
	return imp.errors
}

// Import imports a channel of the given type and specified direction.
// It is equivalent to ImportNValues with a count of -1, meaning unbounded.
func (imp *Importer) Import(name string, chT interface{}, dir Dir) os.Error {
	return imp.ImportNValues(name, chT, dir, -1)
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
//	err = imp.ImportNValues("name", ch, Recv, 1)
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
	imp.chans[name] = &chanDir{ch, dir}
	// Tell the other side about this channel.
	hdr := &header{name: name}
	req := &request{count: int64(n), dir: dir}
	if err = imp.encode(hdr, payRequest, req); err != nil {
		impLog("request encode:", err)
		return err
	}
	if dir == Send {
		go func() {
			for i := 0; n == -1 || i < n; i++ {
				val := ch.Recv()
				if ch.Closed() {
					if err = imp.encode(hdr, payClosed, nil); err != nil {
						impLog("error encoding client closed message:", err)
					}
					return
				}
				if err = imp.encode(hdr, payData, val.Interface()); err != nil {
					impLog("error encoding client send:", err)
					return
				}
			}
		}()
	}
	return nil
}

// Hangup disassociates the named channel from the Importer and closes
// the channel.  Messages in flight for the channel may be dropped.
func (imp *Importer) Hangup(name string) os.Error {
	imp.chanLock.Lock()
	chDir, ok := imp.chans[name]
	if ok {
		imp.chans[name] = nil, false
	}
	imp.chanLock.Unlock()
	if !ok {
		return os.ErrorString("netchan import: hangup: no such channel: " + name)
	}
	chDir.ch.Close()
	return nil
}
