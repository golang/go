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

type Decoder struct {
	sync.Mutex;	// each item must be received atomically
	r	io.Reader;	// source of the data
	seen	map[TypeId] *wireType;	// which types we've already seen described
	state	*decodeState;	// reads data from in-memory buffer
	countState	*decodeState;	// reads counts from wire
	buf	[]byte;
	oneByte	[]byte;
}

func NewDecoder(r io.Reader) *Decoder {
	dec := new(Decoder);
	dec.r = r;
	dec.seen = make(map[TypeId] *wireType);
	dec.state = new(decodeState);	// buffer set in Decode(); rest is unimportant
	dec.oneByte = make([]byte, 1);

	return dec;
}

func (dec *Decoder) recvType(id TypeId) {
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

// The value underlying e must be the correct type for the next
// value to be received for this decoder.
func (dec *Decoder) Decode(e interface{}) os.Error {
	rt, indir := indirect(reflect.Typeof(e));

	// Make sure we're single-threaded through here.
	dec.Lock();
	defer dec.Unlock();

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
		id := TypeId(decodeInt(dec.state));
		if dec.state.err != nil {
			return dec.state.err
		}

		// Is it a new type?
		if id < 0 {	// 0 is the error state, handled above
			// If the id is negative, we have a type.
			dec.recvType(-id);
			if dec.state.err != nil {
				return dec.state.err
			}
			continue;
		}

		// No, it's a value.
		typeLock.Lock();
		info := getTypeInfo(rt);
		typeLock.Unlock();

		// Check type compatibility.
		// TODO(r): need to make the decoder work correctly if the wire type is compatible
		// but not equal to the local type (e.g, extra fields).
		if info.wire.name() != dec.seen[id].name() {
			dec.state.err = os.ErrorString("gob decode: incorrect type for wire value: want " + info.wire.name() + "; received " + dec.seen[id].name());
			return dec.state.err
		}

		// Receive a value.
		decode(dec.state.b, id, e);

		return dec.state.err
	}
	return nil	// silence compiler
}
