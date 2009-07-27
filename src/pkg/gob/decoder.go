// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes";
	"gob";
	"io";
	"os";
	"reflect";
	"sync";
)

// A Decoder manages the receipt of type and data information read from the
// remote side of a connection.
type Decoder struct {
	mutex	sync.Mutex;	// each item must be received atomically
	r	io.Reader;	// source of the data
	seen	map[typeId] *wireType;	// which types we've already seen described
	state	*decodeState;	// reads data from in-memory buffer
	countState	*decodeState;	// reads counts from wire
	buf	[]byte;
	oneByte	[]byte;
}

// NewDecoder returns a new decoder that reads from the io.Reader.
func NewDecoder(r io.Reader) *Decoder {
	dec := new(Decoder);
	dec.r = r;
	dec.seen = make(map[typeId] *wireType);
	dec.state = new(decodeState);	// buffer set in Decode(); rest is unimportant
	dec.oneByte = make([]byte, 1);

	return dec;
}

func (dec *Decoder) recvType(id typeId) {
	// Have we already seen this type?  That's an error
	if wt_, alreadySeen := dec.seen[id]; alreadySeen {
		dec.state.err = os.ErrorString("gob: duplicate type received");
		return
	}

	// Type:
	wire := new(wireType);
	decode(dec.state.b, tWireType, wire);
	// Remember we've seen this type.
	dec.seen[id] = wire;
}

// Decode reads the next value from the connection and stores
// it in the data represented by the empty interface value.
// The value underlying e must be the correct type for the next
// data item received.
func (dec *Decoder) Decode(e interface{}) os.Error {
	rt, indir := indirect(reflect.Typeof(e));

	// Make sure we're single-threaded through here.
	dec.mutex.Lock();
	defer dec.mutex.Unlock();

	dec.state.err = nil;
	for {
		// Read a count.
		nbytes, err := decodeUintReader(dec.r, dec.oneByte);
		if err != nil {
			return err;
		}

		// Allocate the buffer.
		if nbytes > uint64(len(dec.buf)) {
			dec.buf = make([]byte, nbytes + 1000);
		}
		dec.state.b = bytes.NewBuffer(dec.buf[0:nbytes]);

		// Read the data
		var n int;
		n, err = dec.r.Read(dec.buf[0:nbytes]);
		if err != nil {
			return err;
		}
		if n < int(nbytes) {
			return os.ErrorString("gob decode: short read");
		}

		// Receive a type id.
		id := typeId(decodeInt(dec.state));
		if dec.state.err != nil {
			break;
		}

		// Is it a new type?
		if id < 0 {	// 0 is the error state, handled above
			// If the id is negative, we have a type.
			dec.recvType(-id);
			if dec.state.err != nil {
				break;
			}
			continue;
		}

		// No, it's a value.
		dec.state.err = decode(dec.state.b, id, e);
		break;
	}
	return dec.state.err
}
