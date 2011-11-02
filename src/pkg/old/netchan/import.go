// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"errors"
	"io"
	"log"
	"net"
	"reflect"
	"sync"
	"time"
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
	chanLock sync.Mutex // protects access to channel map
	names    map[string]*netChan
	chans    map[int]*netChan
	errors   chan error
	maxId    int
	mu       sync.Mutex // protects remaining fields
	unacked  int64      // number of unacknowledged sends.
	seqLock  sync.Mutex // guarantees messages are in sequence, only locked under mu
}

// NewImporter creates a new Importer object to import a set of channels
// from the given connection. The Exporter must be available and serving when
// the Importer is created.
func NewImporter(conn io.ReadWriter) *Importer {
	imp := new(Importer)
	imp.encDec = newEncDec(conn)
	imp.chans = make(map[int]*netChan)
	imp.names = make(map[string]*netChan)
	imp.errors = make(chan error, 10)
	imp.unacked = 0
	go imp.run()
	return imp
}

// Import imports a set of channels from the given network and address.
func Import(network, remoteaddr string) (*Importer, error) {
	conn, err := net.Dial(network, remoteaddr)
	if err != nil {
		return nil, err
	}
	return NewImporter(conn), nil
}

// shutdown closes all channels for which we are receiving data from the remote side.
func (imp *Importer) shutdown() {
	imp.chanLock.Lock()
	for _, ich := range imp.chans {
		if ich.dir == Recv {
			ich.close()
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
	hdrValue := reflect.ValueOf(hdr)
	ackHdr := new(header)
	err := new(error_)
	errValue := reflect.ValueOf(err)
	for {
		*hdr = header{}
		if e := imp.decode(hdrValue); e != nil {
			if e != io.EOF {
				impLog("header:", e)
				imp.shutdown()
			}
			return
		}
		switch hdr.PayloadType {
		case payData:
			// done lower in loop
		case payError:
			if e := imp.decode(errValue); e != nil {
				impLog("error:", e)
				return
			}
			if err.Error != "" {
				impLog("response error:", err.Error)
				select {
				case imp.errors <- errors.New(err.Error):
					continue // errors are not acknowledged
				default:
					imp.shutdown()
					return
				}
			}
		case payClosed:
			nch := imp.getChan(hdr.Id, false)
			if nch != nil {
				nch.close()
			}
			continue // closes are not acknowledged.
		case payAckSend:
			// we can receive spurious acks if the channel is
			// hung up, so we ask getChan to ignore any errors.
			nch := imp.getChan(hdr.Id, true)
			if nch != nil {
				nch.acked()
				imp.mu.Lock()
				imp.unacked--
				imp.mu.Unlock()
			}
			continue
		default:
			impLog("unexpected payload type:", hdr.PayloadType)
			return
		}
		nch := imp.getChan(hdr.Id, false)
		if nch == nil {
			continue
		}
		if nch.dir != Recv {
			impLog("cannot happen: receive from non-Recv channel")
			return
		}
		// Acknowledge receipt
		ackHdr.Id = hdr.Id
		ackHdr.SeqNum = hdr.SeqNum
		imp.encode(ackHdr, payAck, nil)
		// Create a new value for each received item.
		value := reflect.New(nch.ch.Type().Elem()).Elem()
		if e := imp.decode(value); e != nil {
			impLog("importer value decode:", e)
			return
		}
		nch.send(value)
	}
}

func (imp *Importer) getChan(id int, errOk bool) *netChan {
	imp.chanLock.Lock()
	ich := imp.chans[id]
	imp.chanLock.Unlock()
	if ich == nil {
		if !errOk {
			impLog("unknown id in netchan request: ", id)
		}
		return nil
	}
	return ich
}

// Errors returns a channel from which transmission and protocol errors
// can be read. Clients of the importer are not required to read the error
// channel for correct execution. However, if too many errors occur
// without being read from the error channel, the importer will shut down.
func (imp *Importer) Errors() chan error {
	return imp.errors
}

// Import imports a channel of the given type, size and specified direction.
// It is equivalent to ImportNValues with a count of -1, meaning unbounded.
func (imp *Importer) Import(name string, chT interface{}, dir Dir, size int) error {
	return imp.ImportNValues(name, chT, dir, size, -1)
}

// ImportNValues imports a channel of the given type and specified
// direction and then receives or transmits up to n values on that
// channel.  A value of n==-1 implies an unbounded number of values.  The
// channel will have buffer space for size values, or 1 value if size < 1.
// The channel to be bound to the remote site's channel is provided
// in the call and may be of arbitrary channel type.
// Despite the literal signature, the effective signature is
//	ImportNValues(name string, chT chan T, dir Dir, size, n int) error
// Example usage:
//	imp, err := NewImporter("tcp", "netchanserver.mydomain.com:1234")
//	if err != nil { log.Fatal(err) }
//	ch := make(chan myType)
//	err = imp.ImportNValues("name", ch, Recv, 1, 1)
//	if err != nil { log.Fatal(err) }
//	fmt.Printf("%+v\n", <-ch)
func (imp *Importer) ImportNValues(name string, chT interface{}, dir Dir, size, n int) error {
	ch, err := checkChan(chT, dir)
	if err != nil {
		return err
	}
	imp.chanLock.Lock()
	defer imp.chanLock.Unlock()
	_, present := imp.names[name]
	if present {
		return errors.New("channel name already being imported:" + name)
	}
	if size < 1 {
		size = 1
	}
	id := imp.maxId
	imp.maxId++
	nch := newNetChan(name, id, &chanDir{ch, dir}, imp.encDec, size, int64(n))
	imp.names[name] = nch
	imp.chans[id] = nch
	// Tell the other side about this channel.
	hdr := &header{Id: id}
	req := &request{Name: name, Count: int64(n), Dir: dir, Size: size}
	if err = imp.encode(hdr, payRequest, req); err != nil {
		impLog("request encode:", err)
		return err
	}
	if dir == Send {
		go func() {
			for i := 0; n == -1 || i < n; i++ {
				val, ok := nch.recv()
				if !ok {
					if err = imp.encode(hdr, payClosed, nil); err != nil {
						impLog("error encoding client closed message:", err)
					}
					return
				}
				// We hold the lock during transmission to guarantee messages are
				// sent in order.
				imp.mu.Lock()
				imp.unacked++
				imp.seqLock.Lock()
				imp.mu.Unlock()
				if err = imp.encode(hdr, payData, val.Interface()); err != nil {
					impLog("error encoding client send:", err)
					return
				}
				imp.seqLock.Unlock()
			}
		}()
	}
	return nil
}

// Hangup disassociates the named channel from the Importer and closes
// the channel.  Messages in flight for the channel may be dropped.
func (imp *Importer) Hangup(name string) error {
	imp.chanLock.Lock()
	defer imp.chanLock.Unlock()
	nc := imp.names[name]
	if nc == nil {
		return errors.New("netchan import: hangup: no such channel: " + name)
	}
	delete(imp.names, name)
	delete(imp.chans, nc.id)
	nc.close()
	return nil
}

func (imp *Importer) unackedCount() int64 {
	imp.mu.Lock()
	n := imp.unacked
	imp.mu.Unlock()
	return n
}

// Drain waits until all messages sent from this exporter/importer, including
// those not yet sent to any server and possibly including those sent while
// Drain was executing, have been received by the exporter.  In short, it
// waits until all the importer's messages have been received.
// If the timeout (measured in nanoseconds) is positive and Drain takes
// longer than that to complete, an error is returned.
func (imp *Importer) Drain(timeout int64) error {
	startTime := time.Nanoseconds()
	for imp.unackedCount() > 0 {
		if timeout > 0 && time.Nanoseconds()-startTime >= timeout {
			return errors.New("timeout")
		}
		time.Sleep(100 * 1e6)
	}
	return nil
}
