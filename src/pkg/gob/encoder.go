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

type Encoder struct {
	sync.Mutex;	// each item must be sent atomically
	w	io.Writer;	// where to send the data
	sent	map[reflect.Type] TypeId;	// which types we've already sent
	state	*encoderState;	// so we can encode integers, strings directly
	countState	*encoderState;	// stage for writing counts
	buf	[]byte;	// for collecting the output.
}

func NewEncoder(w io.Writer) *Encoder {
	enc := new(Encoder);
	enc.w = w;
	enc.sent = make(map[reflect.Type] TypeId);
	enc.state = new(encoderState);
	enc.state.b = new(bytes.Buffer);	// the rest isn't important; all we need is buffer and writer
	enc.countState = new(encoderState);
	enc.countState.b = new(bytes.Buffer);	// the rest isn't important; all we need is buffer and writer
	return enc;
}

func (enc *Encoder) badType(rt reflect.Type) {
	enc.state.err = os.ErrorString("gob: can't encode type " + rt.String());
}

// Send the data item preceded by a unsigned count of its length.
func (enc *Encoder) send() {
	// Encode the length.
	encodeUint(enc.countState, uint64(enc.state.b.Len()));
	// Build the buffer.
	countLen := enc.countState.b.Len();
	total := countLen + enc.state.b.Len();
	if total > len(enc.buf) {
		enc.buf = make([]byte, total+1000);	// extra for growth
	}
	// Place the length before the data.
	// TODO(r): avoid the extra copy here.
	enc.countState.b.Read(enc.buf[0:countLen]);
	// Now the data.
	enc.state.b.Read(enc.buf[countLen:total]);
	// Write the data.
	enc.w.Write(enc.buf[0:total]);
}

func (enc *Encoder) sendType(origt reflect.Type) {
	// Drill down to the base type.
	rt, indir_ := indirect(origt);

	// We only send structs - everything else is basic or an error
	switch t := rt.(type) {
	case *reflect.StructType:
		break;	// we handle these
	case *reflect.ChanType:
		enc.badType(rt);
		return;
	case *reflect.MapType:
		enc.badType(rt);
		return;
	case *reflect.FuncType:
		enc.badType(rt);
		return;
	case *reflect.InterfaceType:
		enc.badType(rt);
		return;
	default:
		return;	// basic, array, etc; not a type to be sent.
	}

	// Have we already sent this type?  This time we ask about the base type.
	if id_, alreadySent := enc.sent[rt]; alreadySent {
		return
	}

	// Need to send it.
	info := getTypeInfo(rt);
	// Send the pair (-id, type)
	// Id:
	encodeInt(enc.state, -int64(info.typeId));
	// Type:
	encode(enc.state.b, info.wire);
	enc.send();

	// Remember we've sent this type.
	enc.sent[rt] = info.typeId;
	// Remember we've sent the top-level, possibly indirect type too.
	enc.sent[origt] = info.typeId;
	// Now send the inner types
	st := rt.(*reflect.StructType);
	for i := 0; i < st.NumField(); i++ {
		enc.sendType(st.Field(i).Type);
	}
}

func (enc *Encoder) Encode(e interface{}) os.Error {
	if enc.state.b.Len() > 0 || enc.countState.b.Len() > 0 {
		panicln("Encoder: buffer not empty")
	}
	rt, indir := indirect(reflect.Typeof(e));

	// Make sure we're single-threaded through here.
	enc.Lock();
	defer enc.Unlock();

	// Make sure the type is known to the other side.
	// First, have we already sent this type?
	if id_, alreadySent := enc.sent[rt]; !alreadySent {
		// No, so send it.
		enc.sendType(rt);
		if enc.state.err != nil {
			enc.state.b.Reset();
			enc.countState.b.Reset();
			return enc.state.err
		}
	}

	// Identify the type of this top-level value.
	encodeInt(enc.state, int64(enc.sent[rt]));

	// Encode the object.
	encode(enc.state.b, e);
	enc.send();

	return enc.state.err
}
