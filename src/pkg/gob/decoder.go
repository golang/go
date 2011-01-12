// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"io"
	"os"
	"reflect"
	"sync"
)

// A Decoder manages the receipt of type and data information read from the
// remote side of a connection.
type Decoder struct {
	mutex        sync.Mutex                              // each item must be received atomically
	r            io.Reader                               // source of the data
	wireType     map[typeId]*wireType                    // map from remote ID to local description
	decoderCache map[reflect.Type]map[typeId]**decEngine // cache of compiled engines
	ignorerCache map[typeId]**decEngine                  // ditto for ignored objects
	state        *decodeState                            // reads data from in-memory buffer
	countState   *decodeState                            // reads counts from wire
	buf          []byte
	countBuf     [9]byte // counts may be uint64s (unlikely!), require 9 bytes
	byteBuffer   *bytes.Buffer
	err          os.Error
}

// NewDecoder returns a new decoder that reads from the io.Reader.
func NewDecoder(r io.Reader) *Decoder {
	dec := new(Decoder)
	dec.r = r
	dec.wireType = make(map[typeId]*wireType)
	dec.state = newDecodeState(dec, &dec.byteBuffer) // buffer set in Decode()
	dec.decoderCache = make(map[reflect.Type]map[typeId]**decEngine)
	dec.ignorerCache = make(map[typeId]**decEngine)

	return dec
}

// recvType loads the definition of a type and reloads the Decoder's buffer.
func (dec *Decoder) recvType(id typeId) {
	// Have we already seen this type?  That's an error
	if dec.wireType[id] != nil {
		dec.err = os.ErrorString("gob: duplicate type received")
		return
	}

	// Type:
	wire := new(wireType)
	dec.err = dec.decode(tWireType, reflect.NewValue(wire))
	if dec.err != nil {
		return
	}
	// Remember we've seen this type.
	dec.wireType[id] = wire

	// Load the next parcel.
	dec.recv()
}

// Decode reads the next value from the connection and stores
// it in the data represented by the empty interface value.
// The value underlying e must be the correct type for the next
// data item received, and must be a pointer.
func (dec *Decoder) Decode(e interface{}) os.Error {
	value := reflect.NewValue(e)
	// If e represents a value as opposed to a pointer, the answer won't
	// get back to the caller.  Make sure it's a pointer.
	if value.Type().Kind() != reflect.Ptr {
		dec.err = os.ErrorString("gob: attempt to decode into a non-pointer")
		return dec.err
	}
	return dec.DecodeValue(value)
}

// recv reads the next count-delimited item from the input. It is the converse
// of Encoder.send.
func (dec *Decoder) recv() {
	// Read a count.
	var nbytes uint64
	nbytes, dec.err = decodeUintReader(dec.r, dec.countBuf[0:])
	if dec.err != nil {
		return
	}
	// Allocate the buffer.
	if nbytes > uint64(len(dec.buf)) {
		dec.buf = make([]byte, nbytes+1000)
	}
	dec.byteBuffer = bytes.NewBuffer(dec.buf[0:nbytes])

	// Read the data
	_, dec.err = io.ReadFull(dec.r, dec.buf[0:nbytes])
	if dec.err != nil {
		if dec.err == os.EOF {
			dec.err = io.ErrUnexpectedEOF
		}
		return
	}
}

// decodeValueFromBuffer grabs the next value from the input. The Decoder's
// buffer already contains data.  If the next item in the buffer is a type
// descriptor, it may be necessary to reload the buffer, but recvType does that.
func (dec *Decoder) decodeValueFromBuffer(value reflect.Value, ignoreInterfaceValue, countPresent bool) {
	for dec.state.b.Len() > 0 {
		// Receive a type id.
		id := typeId(dec.state.decodeInt())

		// Is it a new type?
		if id < 0 { // 0 is the error state, handled above
			// If the id is negative, we have a type.
			dec.recvType(-id)
			if dec.err != nil {
				break
			}
			continue
		}

		// Make sure the type has been defined already or is a builtin type (for
		// top-level singleton values).
		if dec.wireType[id] == nil && builtinIdToType[id] == nil {
			dec.err = errBadType
			break
		}
		// An interface value is preceded by a byte count.
		if countPresent {
			count := int(dec.state.decodeUint())
			if ignoreInterfaceValue {
				// An interface value is preceded by a byte count. Just skip that many bytes.
				dec.state.b.Next(int(count))
				break
			}
			// Otherwise fall through and decode it.
		}
		dec.err = dec.decode(id, value)
		break
	}
}

// DecodeValue reads the next value from the connection and stores
// it in the data represented by the reflection value.
// The value must be the correct type for the next
// data item received.
func (dec *Decoder) DecodeValue(value reflect.Value) os.Error {
	// Make sure we're single-threaded through here.
	dec.mutex.Lock()
	defer dec.mutex.Unlock()

	dec.err = nil
	dec.recv()
	if dec.err != nil {
		return dec.err
	}
	dec.decodeValueFromBuffer(value, false, false)
	return dec.err
}

// If debug.go is compiled into the program , debugFunc prints a human-readable
// representation of the gob data read from r by calling that file's Debug function.
// Otherwise it is nil.
var debugFunc func(io.Reader)
