// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"gob";
	"io";
	"os";
	"reflect";
	"sync";
)

type Decoder struct {
	sync.Mutex;	// each item must be received atomically
	seen	map[TypeId] *wireType;	// which types we've already seen described
	state	*DecState;	// so we can encode integers, strings directly
}

func NewDecoder(r io.Reader) *Decoder {
	dec := new(Decoder);
	dec.seen = make(map[TypeId] *wireType);
	dec.state = new(DecState);
	dec.state.r = r;	// the rest isn't important; all we need is buffer and reader

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
	Decode(dec.state.r, wire);
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

	var id TypeId;
	for dec.state.err == nil {
		// Receive a type id.
		id = TypeId(DecodeInt(dec.state));

		// If the id is positive, we have a value.  0 is the error state
		if id >= 0 {
			break;
		}

		// The id is negative; a type descriptor follows.
		dec.recvType(-id);
	}
	if dec.state.err != nil {
		return dec.state.err
	}

	info := getTypeInfo(rt);

	// Check type compatibility.
	// TODO(r): need to make the decoder work correctly if the wire type is compatible
	// but not equal to the local type (e.g, extra fields).
	if info.wire.name() != dec.seen[id].name() {
		dec.state.err = os.ErrorString("gob decode: incorrect type for wire value");
		return dec.state.err
	}

	// Receive a value.
	Decode(dec.state.r, e);

	// Release and return.
	return dec.state.err
}
