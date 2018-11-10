// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bufio"
	"errors"
	"io"
	"reflect"
	"sync"
)

// tooBig provides a sanity check for sizes; used in several places.
// Upper limit of 1GB, allowing room to grow a little without overflow.
// TODO: make this adjustable?
const tooBig = 1 << 30

// A Decoder manages the receipt of type and data information read from the
// remote side of a connection.
//
// The Decoder does only basic sanity checking on decoded input sizes,
// and its limits are not configurable. Take caution when decoding gob data
// from untrusted sources.
type Decoder struct {
	mutex        sync.Mutex                              // each item must be received atomically
	r            io.Reader                               // source of the data
	buf          decBuffer                               // buffer for more efficient i/o from r
	wireType     map[typeId]*wireType                    // map from remote ID to local description
	decoderCache map[reflect.Type]map[typeId]**decEngine // cache of compiled engines
	ignorerCache map[typeId]**decEngine                  // ditto for ignored objects
	freeList     *decoderState                           // list of free decoderStates; avoids reallocation
	countBuf     []byte                                  // used for decoding integers while parsing messages
	err          error
}

// NewDecoder returns a new decoder that reads from the io.Reader.
// If r does not also implement io.ByteReader, it will be wrapped in a
// bufio.Reader.
func NewDecoder(r io.Reader) *Decoder {
	dec := new(Decoder)
	// We use the ability to read bytes as a plausible surrogate for buffering.
	if _, ok := r.(io.ByteReader); !ok {
		r = bufio.NewReader(r)
	}
	dec.r = r
	dec.wireType = make(map[typeId]*wireType)
	dec.decoderCache = make(map[reflect.Type]map[typeId]**decEngine)
	dec.ignorerCache = make(map[typeId]**decEngine)
	dec.countBuf = make([]byte, 9) // counts may be uint64s (unlikely!), require 9 bytes

	return dec
}

// recvType loads the definition of a type.
func (dec *Decoder) recvType(id typeId) {
	// Have we already seen this type?  That's an error
	if id < firstUserId || dec.wireType[id] != nil {
		dec.err = errors.New("gob: duplicate type received")
		return
	}

	// Type:
	wire := new(wireType)
	dec.decodeValue(tWireType, reflect.ValueOf(wire))
	if dec.err != nil {
		return
	}
	// Remember we've seen this type.
	dec.wireType[id] = wire
}

var errBadCount = errors.New("invalid message length")

// recvMessage reads the next count-delimited item from the input. It is the converse
// of Encoder.writeMessage. It returns false on EOF or other error reading the message.
func (dec *Decoder) recvMessage() bool {
	// Read a count.
	nbytes, _, err := decodeUintReader(dec.r, dec.countBuf)
	if err != nil {
		dec.err = err
		return false
	}
	if nbytes >= tooBig {
		dec.err = errBadCount
		return false
	}
	dec.readMessage(int(nbytes))
	return dec.err == nil
}

// readMessage reads the next nbytes bytes from the input.
func (dec *Decoder) readMessage(nbytes int) {
	if dec.buf.Len() != 0 {
		// The buffer should always be empty now.
		panic("non-empty decoder buffer")
	}
	// Read the data
	dec.buf.Size(nbytes)
	_, dec.err = io.ReadFull(dec.r, dec.buf.Bytes())
	if dec.err != nil {
		if dec.err == io.EOF {
			dec.err = io.ErrUnexpectedEOF
		}
	}
}

// toInt turns an encoded uint64 into an int, according to the marshaling rules.
func toInt(x uint64) int64 {
	i := int64(x >> 1)
	if x&1 != 0 {
		i = ^i
	}
	return i
}

func (dec *Decoder) nextInt() int64 {
	n, _, err := decodeUintReader(&dec.buf, dec.countBuf)
	if err != nil {
		dec.err = err
	}
	return toInt(n)
}

func (dec *Decoder) nextUint() uint64 {
	n, _, err := decodeUintReader(&dec.buf, dec.countBuf)
	if err != nil {
		dec.err = err
	}
	return n
}

// decodeTypeSequence parses:
// TypeSequence
//	(TypeDefinition DelimitedTypeDefinition*)?
// and returns the type id of the next value. It returns -1 at
// EOF.  Upon return, the remainder of dec.buf is the value to be
// decoded. If this is an interface value, it can be ignored by
// resetting that buffer.
func (dec *Decoder) decodeTypeSequence(isInterface bool) typeId {
	for dec.err == nil {
		if dec.buf.Len() == 0 {
			if !dec.recvMessage() {
				break
			}
		}
		// Receive a type id.
		id := typeId(dec.nextInt())
		if id >= 0 {
			// Value follows.
			return id
		}
		// Type definition for (-id) follows.
		dec.recvType(-id)
		// When decoding an interface, after a type there may be a
		// DelimitedValue still in the buffer. Skip its count.
		// (Alternatively, the buffer is empty and the byte count
		// will be absorbed by recvMessage.)
		if dec.buf.Len() > 0 {
			if !isInterface {
				dec.err = errors.New("extra data in buffer")
				break
			}
			dec.nextUint()
		}
	}
	return -1
}

// Decode reads the next value from the input stream and stores
// it in the data represented by the empty interface value.
// If e is nil, the value will be discarded. Otherwise,
// the value underlying e must be a pointer to the
// correct type for the next data item received.
// If the input is at EOF, Decode returns io.EOF and
// does not modify e.
func (dec *Decoder) Decode(e interface{}) error {
	if e == nil {
		return dec.DecodeValue(reflect.Value{})
	}
	value := reflect.ValueOf(e)
	// If e represents a value as opposed to a pointer, the answer won't
	// get back to the caller. Make sure it's a pointer.
	if value.Type().Kind() != reflect.Ptr {
		dec.err = errors.New("gob: attempt to decode into a non-pointer")
		return dec.err
	}
	return dec.DecodeValue(value)
}

// DecodeValue reads the next value from the input stream.
// If v is the zero reflect.Value (v.Kind() == Invalid), DecodeValue discards the value.
// Otherwise, it stores the value into v. In that case, v must represent
// a non-nil pointer to data or be an assignable reflect.Value (v.CanSet())
// If the input is at EOF, DecodeValue returns io.EOF and
// does not modify v.
func (dec *Decoder) DecodeValue(v reflect.Value) error {
	if v.IsValid() {
		if v.Kind() == reflect.Ptr && !v.IsNil() {
			// That's okay, we'll store through the pointer.
		} else if !v.CanSet() {
			return errors.New("gob: DecodeValue of unassignable value")
		}
	}
	// Make sure we're single-threaded through here.
	dec.mutex.Lock()
	defer dec.mutex.Unlock()

	dec.buf.Reset() // In case data lingers from previous invocation.
	dec.err = nil
	id := dec.decodeTypeSequence(false)
	if dec.err == nil {
		dec.decodeValue(id, v)
	}
	return dec.err
}

// If debug.go is compiled into the program , debugFunc prints a human-readable
// representation of the gob data read from r by calling that file's Debug function.
// Otherwise it is nil.
var debugFunc func(io.Reader)
