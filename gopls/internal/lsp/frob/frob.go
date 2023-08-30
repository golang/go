// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package frob is a fast restricted object encoder/decoder in the
// spirit of encoding/gob.
//
// As with gob, types that recursively contain functions, channels,
// and unsafe.Pointers cannot be encoded, but frob has these
// additional restrictions:
//
//   - Interface values are not supported; this avoids the need for
//     the encoding to describe types.
//
//   - Types that recursively contain private struct fields are not
//     permitted.
//
//   - The encoding is unspecified and subject to change, so the encoder
//     and decoder must exactly agree on their implementation and on the
//     definitions of the target types.
//
//   - Lengths (of arrays, slices, and maps) are currently assumed to
//     fit in 32 bits.
//
//   - There is no error handling. All errors are reported by panicking.
//
//   - Values are serialized as trees, not graphs, so shared subgraphs
//     are encoded repeatedly.
//
//   - No attempt is made to detect cyclic data structures.
package frob

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"sync"
)

// A Codec[T] is an immutable encoder and decoder for values of type T.
type Codec[T any] struct{ frob *frob }

// CodecFor[T] returns a codec for values of type T.
// It panics if type T is unsuitable.
func CodecFor[T any]() Codec[T] {
	frobsMu.Lock()
	defer frobsMu.Unlock()
	return Codec[T]{frobFor(reflect.TypeOf((*T)(nil)).Elem())}
}

func (codec Codec[T]) Encode(v T) []byte          { return codec.frob.Encode(v) }
func (codec Codec[T]) Decode(data []byte, ptr *T) { codec.frob.Decode(data, ptr) }

var (
	frobsMu sync.Mutex
	frobs   = make(map[reflect.Type]*frob)
)

// A frob is an encoder/decoder for a specific type.
type frob struct {
	t     reflect.Type
	kind  reflect.Kind
	elems []*frob // elem (array/slice/ptr), key+value (map), fields (struct)
}

// frobFor returns the frob for a particular type.
// Precondition: caller holds frobsMu.
func frobFor(t reflect.Type) *frob {
	fr, ok := frobs[t]
	if !ok {
		fr = &frob{t: t, kind: t.Kind()}
		frobs[t] = fr

		switch fr.kind {
		case reflect.Bool,
			reflect.Int,
			reflect.Int8,
			reflect.Int16,
			reflect.Int32,
			reflect.Int64,
			reflect.Uint,
			reflect.Uint8,
			reflect.Uint16,
			reflect.Uint32,
			reflect.Uint64,
			reflect.Uintptr,
			reflect.Float32,
			reflect.Float64,
			reflect.Complex64,
			reflect.Complex128,
			reflect.String:

		case reflect.Array,
			reflect.Slice,
			reflect.Ptr: // TODO(adonovan): after go1.18, use Pointer
			fr.addElem(fr.t.Elem())

		case reflect.Map:
			fr.addElem(fr.t.Key())
			fr.addElem(fr.t.Elem())

		case reflect.Struct:
			for i := 0; i < fr.t.NumField(); i++ {
				field := fr.t.Field(i)
				if field.PkgPath != "" {
					panic(fmt.Sprintf("unexported field %v", field))
				}
				fr.addElem(field.Type)
			}

		default:
			// chan, func, interface, unsafe.Pointer
			panic(fmt.Sprintf("type %v is not supported by frob", fr.t))
		}
	}
	return fr
}

func (fr *frob) addElem(t reflect.Type) {
	fr.elems = append(fr.elems, frobFor(t))
}

const magic = "frob"

func (fr *frob) Encode(v any) []byte {
	rv := reflect.ValueOf(v)
	if rv.Type() != fr.t {
		panic(fmt.Sprintf("got %v, want %v", rv.Type(), fr.t))
	}
	w := &writer{}
	w.bytes([]byte(magic))
	fr.encode(w, rv)
	if uint64(len(w.data))>>32 != 0 {
		panic("too large") // includes all cases where len doesn't fit in 32 bits
	}
	return w.data
}

// encode appends the encoding of value v, whose type must be fr.t.
func (fr *frob) encode(out *writer, v reflect.Value) {
	switch fr.kind {
	case reflect.Bool:
		var b byte
		if v.Bool() {
			b = 1
		}
		out.uint8(b)
	case reflect.Int:
		out.uint64(uint64(v.Int()))
	case reflect.Int8:
		out.uint8(uint8(v.Int()))
	case reflect.Int16:
		out.uint16(uint16(v.Int()))
	case reflect.Int32:
		out.uint32(uint32(v.Int()))
	case reflect.Int64:
		out.uint64(uint64(v.Int()))
	case reflect.Uint:
		out.uint64(v.Uint())
	case reflect.Uint8:
		out.uint8(uint8(v.Uint()))
	case reflect.Uint16:
		out.uint16(uint16(v.Uint()))
	case reflect.Uint32:
		out.uint32(uint32(v.Uint()))
	case reflect.Uint64:
		out.uint64(v.Uint())
	case reflect.Uintptr:
		out.uint64(uint64(v.Uint()))
	case reflect.Float32:
		out.uint32(math.Float32bits(float32(v.Float())))
	case reflect.Float64:
		out.uint64(math.Float64bits(v.Float()))
	case reflect.Complex64:
		z := complex64(v.Complex())
		out.uint32(uint32(math.Float32bits(real(z))))
		out.uint32(uint32(math.Float32bits(imag(z))))
	case reflect.Complex128:
		z := v.Complex()
		out.uint64(math.Float64bits(real(z)))
		out.uint64(math.Float64bits(imag(z)))

	case reflect.Array:
		len := v.Type().Len()
		elem := fr.elems[0]
		for i := 0; i < len; i++ {
			elem.encode(out, v.Index(i))
		}

	case reflect.Slice:
		len := v.Len()
		out.uint32(uint32(len))
		if len > 0 {
			elem := fr.elems[0]
			if elem.kind == reflect.Uint8 {
				// []byte fast path
				out.bytes(v.Bytes())
			} else {
				for i := 0; i < len; i++ {
					elem.encode(out, v.Index(i))
				}
			}
		}

	case reflect.Map:
		len := v.Len()
		out.uint32(uint32(len))
		if len > 0 {
			kfrob, vfrob := fr.elems[0], fr.elems[1]
			for iter := v.MapRange(); iter.Next(); {
				kfrob.encode(out, iter.Key())
				vfrob.encode(out, iter.Value())
			}
		}

	case reflect.Ptr: // TODO(adonovan): after go1.18, use Pointer
		if v.IsNil() {
			out.uint8(0)
		} else {
			out.uint8(1)
			fr.elems[0].encode(out, v.Elem())
		}

	case reflect.String:
		len := v.Len()
		out.uint32(uint32(len))
		if len > 0 {
			out.data = append(out.data, v.String()...)
		}

	case reflect.Struct:
		for i, elem := range fr.elems {
			elem.encode(out, v.Field(i))
		}

	default:
		panic(fr.t)
	}
}

func (fr *frob) Decode(data []byte, ptr any) {
	rv := reflect.ValueOf(ptr).Elem()
	if rv.Type() != fr.t {
		panic(fmt.Sprintf("got %v, want %v", rv.Type(), fr.t))
	}
	rd := &reader{data}
	if string(rd.bytes(4)) != magic {
		panic("not a frob-encoded message")
	}
	fr.decode(rd, rv)
	if len(rd.data) > 0 {
		panic("surplus bytes")
	}
}

// decode reads from in, decodes a value, and sets addr to it.
// addr must be a zero-initialized addressable variable of type fr.t.
func (fr *frob) decode(in *reader, addr reflect.Value) {
	switch fr.kind {
	case reflect.Bool:
		addr.SetBool(in.uint8() != 0)
	case reflect.Int:
		addr.SetInt(int64(in.uint64()))
	case reflect.Int8:
		addr.SetInt(int64(in.uint8()))
	case reflect.Int16:
		addr.SetInt(int64(in.uint16()))
	case reflect.Int32:
		addr.SetInt(int64(in.uint32()))
	case reflect.Int64:
		addr.SetInt(int64(in.uint64()))
	case reflect.Uint:
		addr.SetUint(in.uint64())
	case reflect.Uint8:
		addr.SetUint(uint64(in.uint8()))
	case reflect.Uint16:
		addr.SetUint(uint64(in.uint16()))
	case reflect.Uint32:
		addr.SetUint(uint64(in.uint32()))
	case reflect.Uint64:
		addr.SetUint(in.uint64())
	case reflect.Uintptr:
		addr.SetUint(in.uint64())
	case reflect.Float32:
		addr.SetFloat(float64(math.Float32frombits(in.uint32())))
	case reflect.Float64:
		addr.SetFloat(math.Float64frombits(in.uint64()))
	case reflect.Complex64:
		addr.SetComplex(complex128(complex(
			math.Float32frombits(in.uint32()),
			math.Float32frombits(in.uint32()),
		)))
	case reflect.Complex128:
		addr.SetComplex(complex(
			math.Float64frombits(in.uint64()),
			math.Float64frombits(in.uint64()),
		))

	case reflect.Array:
		len := fr.t.Len()
		for i := 0; i < len; i++ {
			fr.elems[0].decode(in, addr.Index(i))
		}

	case reflect.Slice:
		len := int(in.uint32())
		if len > 0 {
			elem := fr.elems[0]
			if elem.kind == reflect.Uint8 {
				// []byte fast path
				// (Not addr.SetBytes: we must make a copy.)
				addr.Set(reflect.AppendSlice(addr, reflect.ValueOf(in.bytes(len))))
			} else {
				addr.Set(reflect.MakeSlice(fr.t, len, len))
				for i := 0; i < len; i++ {
					elem.decode(in, addr.Index(i))
				}
			}
		}

	case reflect.Map:
		len := int(in.uint32())
		if len > 0 {
			m := reflect.MakeMapWithSize(fr.t, len)
			addr.Set(m)
			kfrob, vfrob := fr.elems[0], fr.elems[1]
			k := reflect.New(kfrob.t).Elem()
			v := reflect.New(vfrob.t).Elem()
			kzero := reflect.Zero(kfrob.t)
			vzero := reflect.Zero(vfrob.t)
			for i := 0; i < len; i++ {
				// TODO(adonovan): use SetZero from go1.20.
				// k.SetZero()
				// v.SetZero()
				k.Set(kzero)
				v.Set(vzero)
				kfrob.decode(in, k)
				vfrob.decode(in, v)
				m.SetMapIndex(k, v)
			}
		}

	case reflect.Ptr: // TODO(adonovan): after go1.18, use Pointer
		isNil := in.uint8() == 0
		if !isNil {
			ptr := reflect.New(fr.elems[0].t)
			addr.Set(ptr)
			fr.elems[0].decode(in, ptr.Elem())
		}

	case reflect.String:
		len := int(in.uint32())
		if len > 0 {
			addr.SetString(string(in.bytes(len)))
		}

	case reflect.Struct:
		for i, elem := range fr.elems {
			elem.decode(in, addr.Field(i))
		}

	default:
		panic(fr.t)
	}
}

var le = binary.LittleEndian

type reader struct{ data []byte }

func (r *reader) uint8() uint8 {
	v := r.data[0]
	r.data = r.data[1:]
	return v
}

func (r *reader) uint16() uint16 {
	v := le.Uint16(r.data)
	r.data = r.data[2:]
	return v
}

func (r *reader) uint32() uint32 {
	v := le.Uint32(r.data)
	r.data = r.data[4:]
	return v
}

func (r *reader) uint64() uint64 {
	v := le.Uint64(r.data)
	r.data = r.data[8:]
	return v
}

func (r *reader) bytes(n int) []byte {
	v := r.data[:n]
	r.data = r.data[n:]
	return v
}

type writer struct{ data []byte }

func (w *writer) uint8(v uint8)   { w.data = append(w.data, v) }
func (w *writer) uint16(v uint16) { w.data = appendUint16(w.data, v) }
func (w *writer) uint32(v uint32) { w.data = appendUint32(w.data, v) }
func (w *writer) uint64(v uint64) { w.data = appendUint64(w.data, v) }
func (w *writer) bytes(v []byte)  { w.data = append(w.data, v...) }

// TODO(adonovan): delete these as in go1.19 they are methods on LittleEndian:

func appendUint16(b []byte, v uint16) []byte {
	return append(b,
		byte(v),
		byte(v>>8),
	)
}

func appendUint32(b []byte, v uint32) []byte {
	return append(b,
		byte(v),
		byte(v>>8),
		byte(v>>16),
		byte(v>>24),
	)
}

func appendUint64(b []byte, v uint64) []byte {
	return append(b,
		byte(v),
		byte(v>>8),
		byte(v>>16),
		byte(v>>24),
		byte(v>>32),
		byte(v>>40),
		byte(v>>48),
		byte(v>>56),
	)
}
