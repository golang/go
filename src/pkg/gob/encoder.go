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

// An Encoder manages the transmission of type and data information to the
// other side of a connection.
type Encoder struct {
	mutex      sync.Mutex              // each item must be sent atomically
	w          io.Writer               // where to send the data
	sent       map[reflect.Type]typeId // which types we've already sent
	state      *encoderState           // so we can encode integers, strings directly
	countState *encoderState           // stage for writing counts
	buf        []byte                  // for collecting the output.
	err        os.Error
}

// NewEncoder returns a new encoder that will transmit on the io.Writer.
func NewEncoder(w io.Writer) *Encoder {
	enc := new(Encoder)
	enc.w = w
	enc.sent = make(map[reflect.Type]typeId)
	enc.state = newEncoderState(enc, new(bytes.Buffer))
	enc.countState = newEncoderState(enc, new(bytes.Buffer))
	return enc
}

func (enc *Encoder) badType(rt reflect.Type) {
	enc.setError(os.ErrorString("gob: can't encode type " + rt.String()))
}

func (enc *Encoder) setError(err os.Error) {
	if enc.err == nil { // remember the first.
		enc.err = err
	}
	enc.state.b.Reset()
}

// Send the data item preceded by a unsigned count of its length.
func (enc *Encoder) send() {
	// Encode the length.
	enc.countState.encodeUint(uint64(enc.state.b.Len()))
	// Build the buffer.
	countLen := enc.countState.b.Len()
	total := countLen + enc.state.b.Len()
	if total > len(enc.buf) {
		enc.buf = make([]byte, total+1000) // extra for growth
	}
	// Place the length before the data.
	// TODO(r): avoid the extra copy here.
	enc.countState.b.Read(enc.buf[0:countLen])
	// Now the data.
	enc.state.b.Read(enc.buf[countLen:total])
	// Write the data.
	_, err := enc.w.Write(enc.buf[0:total])
	if err != nil {
		enc.setError(err)
	}
}

func (enc *Encoder) sendType(origt reflect.Type) (sent bool) {
	// Drill down to the base type.
	rt, _ := indirect(origt)

	switch rt := rt.(type) {
	default:
		// Basic types and interfaces do not need to be described.
		return
	case *reflect.SliceType:
		// If it's []uint8, don't send; it's considered basic.
		if rt.Elem().Kind() == reflect.Uint8 {
			return
		}
		// Otherwise we do send.
		break
	case *reflect.ArrayType:
		// arrays must be sent so we know their lengths and element types.
		break
	case *reflect.MapType:
		// maps must be sent so we know their lengths and key/value types.
		break
	case *reflect.StructType:
		// structs must be sent so we know their fields.
		break
	case *reflect.ChanType, *reflect.FuncType:
		// Probably a bad field in a struct.
		enc.badType(rt)
		return
	}

	// Have we already sent this type?  This time we ask about the base type.
	if _, alreadySent := enc.sent[rt]; alreadySent {
		return
	}

	// Need to send it.
	typeLock.Lock()
	info, err := getTypeInfo(rt)
	typeLock.Unlock()
	if err != nil {
		enc.setError(err)
		return
	}
	// Send the pair (-id, type)
	// Id:
	enc.state.encodeInt(-int64(info.id))
	// Type:
	enc.encode(enc.state.b, reflect.NewValue(info.wire))
	enc.send()
	if enc.err != nil {
		return
	}

	// Remember we've sent this type.
	enc.sent[rt] = info.id
	// Remember we've sent the top-level, possibly indirect type too.
	enc.sent[origt] = info.id
	// Now send the inner types
	switch st := rt.(type) {
	case *reflect.StructType:
		for i := 0; i < st.NumField(); i++ {
			enc.sendType(st.Field(i).Type)
		}
	case reflect.ArrayOrSliceType:
		enc.sendType(st.Elem())
	}
	return true
}

// Encode transmits the data item represented by the empty interface value,
// guaranteeing that all necessary type information has been transmitted first.
func (enc *Encoder) Encode(e interface{}) os.Error {
	return enc.EncodeValue(reflect.NewValue(e))
}

// sendTypeId makes sure the remote side knows about this type.
// It will send a descriptor if this is the first time the type has been
// sent.  Regardless, it sends the id.
func (enc *Encoder) sendTypeDescriptor(rt reflect.Type) {
	// Make sure the type is known to the other side.
	// First, have we already sent this type?
	if _, alreadySent := enc.sent[rt]; !alreadySent {
		// No, so send it.
		sent := enc.sendType(rt)
		if enc.err != nil {
			return
		}
		// If the type info has still not been transmitted, it means we have
		// a singleton basic type (int, []byte etc.) at top level.  We don't
		// need to send the type info but we do need to update enc.sent.
		if !sent {
			typeLock.Lock()
			info, err := getTypeInfo(rt)
			typeLock.Unlock()
			if err != nil {
				enc.setError(err)
				return
			}
			enc.sent[rt] = info.id
		}
	}

	// Identify the type of this top-level value.
	enc.state.encodeInt(int64(enc.sent[rt]))
}

// EncodeValue transmits the data item represented by the reflection value,
// guaranteeing that all necessary type information has been transmitted first.
func (enc *Encoder) EncodeValue(value reflect.Value) os.Error {
	// Make sure we're single-threaded through here, so multiple
	// goroutines can share an encoder.
	enc.mutex.Lock()
	defer enc.mutex.Unlock()

	enc.err = nil
	rt, _ := indirect(value.Type())

	// Sanity check only: encoder should never come in with data present.
	if enc.state.b.Len() > 0 || enc.countState.b.Len() > 0 {
		enc.err = os.ErrorString("encoder: buffer not empty")
		return enc.err
	}

	enc.sendTypeDescriptor(rt)
	if enc.err != nil {
		return enc.err
	}

	// Encode the object.
	err := enc.encode(enc.state.b, value)
	if err != nil {
		enc.setError(err)
	} else {
		enc.send()
	}

	return enc.err
}
