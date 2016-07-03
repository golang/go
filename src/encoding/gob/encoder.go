// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"errors"
	"io"
	"reflect"
	"sync"
)

// An Encoder manages the transmission of type and data information to the
// other side of a connection.
type Encoder struct {
	mutex      sync.Mutex              // each item must be sent atomically
	w          []io.Writer             // where to send the data
	sent       map[reflect.Type]typeId // which types we've already sent
	countState *encoderState           // stage for writing counts
	freeList   *encoderState           // list of free encoderStates; avoids reallocation
	byteBuf    encBuffer               // buffer for top-level encoderState
	err        error
}

// Before we encode a message, we reserve space at the head of the
// buffer in which to encode its length. This means we can use the
// buffer to assemble the message without another allocation.
const maxLength = 9 // Maximum size of an encoded length.
var spaceForLength = make([]byte, maxLength)

// NewEncoder returns a new encoder that will transmit on the io.Writer.
func NewEncoder(w io.Writer) *Encoder {
	enc := new(Encoder)
	enc.w = []io.Writer{w}
	enc.sent = make(map[reflect.Type]typeId)
	enc.countState = enc.newEncoderState(new(encBuffer))
	return enc
}

// writer() returns the innermost writer the encoder is using
func (enc *Encoder) writer() io.Writer {
	return enc.w[len(enc.w)-1]
}

// pushWriter adds a writer to the encoder.
func (enc *Encoder) pushWriter(w io.Writer) {
	enc.w = append(enc.w, w)
}

// popWriter pops the innermost writer.
func (enc *Encoder) popWriter() {
	enc.w = enc.w[0 : len(enc.w)-1]
}

func (enc *Encoder) setError(err error) {
	if enc.err == nil { // remember the first.
		enc.err = err
	}
}

// writeMessage sends the data item preceded by a unsigned count of its length.
func (enc *Encoder) writeMessage(w io.Writer, b *encBuffer) {
	// Space has been reserved for the length at the head of the message.
	// This is a little dirty: we grab the slice from the bytes.Buffer and massage
	// it by hand.
	message := b.Bytes()
	messageLen := len(message) - maxLength
	// Length cannot be bigger than the decoder can handle.
	if messageLen >= tooBig {
		enc.setError(errors.New("gob: encoder: message too big"))
		return
	}
	// Encode the length.
	enc.countState.b.Reset()
	enc.countState.encodeUint(uint64(messageLen))
	// Copy the length to be a prefix of the message.
	offset := maxLength - enc.countState.b.Len()
	copy(message[offset:], enc.countState.b.Bytes())
	// Write the data.
	_, err := w.Write(message[offset:])
	// Drain the buffer and restore the space at the front for the count of the next message.
	b.Reset()
	b.Write(spaceForLength)
	if err != nil {
		enc.setError(err)
	}
}

// sendActualType sends the requested type, without further investigation, unless
// it's been sent before.
func (enc *Encoder) sendActualType(w io.Writer, state *encoderState, ut *userTypeInfo, actual reflect.Type) (sent bool) {
	if _, alreadySent := enc.sent[actual]; alreadySent {
		return false
	}
	info, err := getTypeInfo(ut)
	if err != nil {
		enc.setError(err)
		return
	}
	// Send the pair (-id, type)
	// Id:
	state.encodeInt(-int64(info.id))
	// Type:
	enc.encode(state.b, reflect.ValueOf(info.wire), wireTypeUserInfo)
	enc.writeMessage(w, state.b)
	if enc.err != nil {
		return
	}

	// Remember we've sent this type, both what the user gave us and the base type.
	enc.sent[ut.base] = info.id
	if ut.user != ut.base {
		enc.sent[ut.user] = info.id
	}
	// Now send the inner types
	switch st := actual; st.Kind() {
	case reflect.Struct:
		for i := 0; i < st.NumField(); i++ {
			if isExported(st.Field(i).Name) {
				enc.sendType(w, state, st.Field(i).Type)
			}
		}
	case reflect.Array, reflect.Slice:
		enc.sendType(w, state, st.Elem())
	case reflect.Map:
		enc.sendType(w, state, st.Key())
		enc.sendType(w, state, st.Elem())
	}
	return true
}

// sendType sends the type info to the other side, if necessary.
func (enc *Encoder) sendType(w io.Writer, state *encoderState, origt reflect.Type) (sent bool) {
	ut := userType(origt)
	if ut.externalEnc != 0 {
		// The rules are different: regardless of the underlying type's representation,
		// we need to tell the other side that the base type is a GobEncoder.
		return enc.sendActualType(w, state, ut, ut.base)
	}

	// It's a concrete value, so drill down to the base type.
	switch rt := ut.base; rt.Kind() {
	default:
		// Basic types and interfaces do not need to be described.
		return
	case reflect.Slice:
		// If it's []uint8, don't send; it's considered basic.
		if rt.Elem().Kind() == reflect.Uint8 {
			return
		}
		// Otherwise we do send.
		break
	case reflect.Array:
		// arrays must be sent so we know their lengths and element types.
		break
	case reflect.Map:
		// maps must be sent so we know their lengths and key/value types.
		break
	case reflect.Struct:
		// structs must be sent so we know their fields.
		break
	case reflect.Chan, reflect.Func:
		// If we get here, it's a field of a struct; ignore it.
		return
	}

	return enc.sendActualType(w, state, ut, ut.base)
}

// Encode transmits the data item represented by the empty interface value,
// guaranteeing that all necessary type information has been transmitted first.
// Passing a nil pointer to Encoder will panic, as they cannot be transmitted by gob.
func (enc *Encoder) Encode(e interface{}) error {
	return enc.EncodeValue(reflect.ValueOf(e))
}

// sendTypeDescriptor makes sure the remote side knows about this type.
// It will send a descriptor if this is the first time the type has been
// sent.
func (enc *Encoder) sendTypeDescriptor(w io.Writer, state *encoderState, ut *userTypeInfo) {
	// Make sure the type is known to the other side.
	// First, have we already sent this type?
	rt := ut.base
	if ut.externalEnc != 0 {
		rt = ut.user
	}
	if _, alreadySent := enc.sent[rt]; !alreadySent {
		// No, so send it.
		sent := enc.sendType(w, state, rt)
		if enc.err != nil {
			return
		}
		// If the type info has still not been transmitted, it means we have
		// a singleton basic type (int, []byte etc.) at top level. We don't
		// need to send the type info but we do need to update enc.sent.
		if !sent {
			info, err := getTypeInfo(ut)
			if err != nil {
				enc.setError(err)
				return
			}
			enc.sent[rt] = info.id
		}
	}
}

// sendTypeId sends the id, which must have already been defined.
func (enc *Encoder) sendTypeId(state *encoderState, ut *userTypeInfo) {
	// Identify the type of this top-level value.
	state.encodeInt(int64(enc.sent[ut.base]))
}

// EncodeValue transmits the data item represented by the reflection value,
// guaranteeing that all necessary type information has been transmitted first.
// Passing a nil pointer to EncodeValue will panic, as they cannot be transmitted by gob.
func (enc *Encoder) EncodeValue(value reflect.Value) error {
	if value.Kind() == reflect.Ptr && value.IsNil() {
		panic("gob: cannot encode nil pointer of type " + value.Type().String())
	}

	// Make sure we're single-threaded through here, so multiple
	// goroutines can share an encoder.
	enc.mutex.Lock()
	defer enc.mutex.Unlock()

	// Remove any nested writers remaining due to previous errors.
	enc.w = enc.w[0:1]

	ut, err := validUserType(value.Type())
	if err != nil {
		return err
	}

	enc.err = nil
	enc.byteBuf.Reset()
	enc.byteBuf.Write(spaceForLength)
	state := enc.newEncoderState(&enc.byteBuf)

	enc.sendTypeDescriptor(enc.writer(), state, ut)
	enc.sendTypeId(state, ut)
	if enc.err != nil {
		return enc.err
	}

	// Encode the object.
	enc.encode(state.b, value, ut)
	if enc.err == nil {
		enc.writeMessage(enc.writer(), state.b)
	}

	enc.freeEncoderState(state)
	return enc.err
}
