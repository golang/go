// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asn1

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"time"
)

// A forkableWriter is an in-memory buffer that can be
// 'forked' to create new forkableWriters that bracket the
// original.  After
//    pre, post := w.fork();
// the overall sequence of bytes represented is logically w+pre+post.
type forkableWriter struct {
	*bytes.Buffer
	pre, post *forkableWriter
}

func newForkableWriter() *forkableWriter {
	return &forkableWriter{bytes.NewBuffer(nil), nil, nil}
}

func (f *forkableWriter) fork() (pre, post *forkableWriter) {
	if f.pre != nil || f.post != nil {
		panic("have already forked")
	}
	f.pre = newForkableWriter()
	f.post = newForkableWriter()
	return f.pre, f.post
}

func (f *forkableWriter) Len() (l int) {
	l += f.Buffer.Len()
	if f.pre != nil {
		l += f.pre.Len()
	}
	if f.post != nil {
		l += f.post.Len()
	}
	return
}

func (f *forkableWriter) writeTo(out io.Writer) (n int, err os.Error) {
	n, err = out.Write(f.Bytes())
	if err != nil {
		return
	}

	var nn int

	if f.pre != nil {
		nn, err = f.pre.writeTo(out)
		n += nn
		if err != nil {
			return
		}
	}

	if f.post != nil {
		nn, err = f.post.writeTo(out)
		n += nn
	}
	return
}

func marshalBase128Int(out *forkableWriter, i int64) (err os.Error) {
	if i == 0 {
		err = out.WriteByte(0)
		return
	}

	for i > 0 {
		next := i >> 7
		o := byte(i & 0x7f)
		if next > 0 {
			o |= 0x80
		}
		err = out.WriteByte(o)
		if err != nil {
			return
		}
		i = next
	}

	return nil
}

func base128Length(i int) (numBytes int) {
	if i == 0 {
		return 1
	}

	for i > 0 {
		numBytes++
		i >>= 7
	}

	return
}

func marshalTagAndLength(out *forkableWriter, t tagAndLength) (err os.Error) {
	b := uint8(t.class) << 6
	if t.isCompound {
		b |= 0x20
	}
	if t.tag >= 31 {
		b |= 0x1f
		err = out.WriteByte(b)
		if err != nil {
			return
		}
		err = marshalBase128Int(out, int64(t.tag))
		if err != nil {
			return
		}
	} else {
		b |= uint8(t.tag)
		err = out.WriteByte(b)
		if err != nil {
			return
		}
	}

	if t.length >= 128 {
		err = out.WriteByte(byte(base128Length(t.length)))
		if err != nil {
			return
		}
		err = marshalBase128Int(out, int64(t.length))
		if err != nil {
			return
		}
	} else {
		err = out.WriteByte(byte(t.length))
		if err != nil {
			return
		}
	}

	return nil
}

func marshalBitString(out *forkableWriter, b BitString) (err os.Error) {
	paddingBits := byte((8 - b.BitLength%8) % 8)
	err = out.WriteByte(paddingBits)
	if err != nil {
		return
	}
	_, err = out.Write(b.Bytes)
	return
}

func marshalObjectIdentifier(out *forkableWriter, oid []int) (err os.Error) {
	if len(oid) < 2 || oid[0] > 6 || oid[1] >= 40 {
		return StructuralError{"invalid object identifier"}
	}

	err = out.WriteByte(byte(oid[0]*40 + oid[1]))
	if err != nil {
		return
	}
	for i := 2; i < len(oid); i++ {
		err = marshalBase128Int(out, int64(oid[i]))
		if err != nil {
			return
		}
	}

	return
}

func marshalPrintableString(out *forkableWriter, s string) (err os.Error) {
	b := strings.Bytes(s)
	for _, c := range b {
		if !isPrintable(c) {
			return StructuralError{"PrintableString contains invalid character"}
		}
	}

	_, err = out.Write(b)
	return
}

func marshalIA5String(out *forkableWriter, s string) (err os.Error) {
	b := strings.Bytes(s)
	for _, c := range b {
		if c > 127 {
			return StructuralError{"IA5String contains invalid character"}
		}
	}

	_, err = out.Write(b)
	return
}

func marshalTwoDigits(out *forkableWriter, v int) (err os.Error) {
	err = out.WriteByte(byte('0' + (v/10)%10))
	if err != nil {
		return
	}
	return out.WriteByte(byte('0' + v%10))
}

func marshalUTCTime(out *forkableWriter, t *time.Time) (err os.Error) {
	switch {
	case 1950 <= t.Year && t.Year < 2000:
		err = marshalTwoDigits(out, int(t.Year-1900))
	case 2000 <= t.Year && t.Year < 2050:
		err = marshalTwoDigits(out, int(t.Year-2000))
	default:
		return StructuralError{"Cannot represent time as UTCTime"}
	}

	if err != nil {
		return
	}

	err = marshalTwoDigits(out, t.Month)
	if err != nil {
		return
	}

	err = marshalTwoDigits(out, t.Day)
	if err != nil {
		return
	}

	err = marshalTwoDigits(out, t.Hour)
	if err != nil {
		return
	}

	err = marshalTwoDigits(out, t.Minute)
	if err != nil {
		return
	}

	err = marshalTwoDigits(out, t.Second)
	if err != nil {
		return
	}

	switch {
	case t.ZoneOffset/60 == 0:
		err = out.WriteByte('Z')
		return
	case t.ZoneOffset > 0:
		err = out.WriteByte('+')
	case t.ZoneOffset < 0:
		err = out.WriteByte('-')
	}

	if err != nil {
		return
	}

	offsetMinutes := t.ZoneOffset / 60
	if offsetMinutes < 0 {
		offsetMinutes = -offsetMinutes
	}

	err = marshalTwoDigits(out, offsetMinutes/60)
	if err != nil {
		return
	}

	err = marshalTwoDigits(out, offsetMinutes%60)
	return
}

func marshalBody(out *forkableWriter, value reflect.Value, params fieldParameters) (err os.Error) {
	switch value.Type() {
	case timeType:
		return marshalUTCTime(out, value.Interface().(*time.Time))
	case bitStringType:
		return marshalBitString(out, value.Interface().(BitString))
	case objectIdentifierType:
		return marshalObjectIdentifier(out, value.Interface().(ObjectIdentifier))
	}

	switch v := value.(type) {
	case *reflect.BoolValue:
		if v.Get() {
			return out.WriteByte(1)
		} else {
			return out.WriteByte(0)
		}
	case *reflect.IntValue:
		return marshalBase128Int(out, int64(v.Get()))
	case *reflect.Int64Value:
		return marshalBase128Int(out, v.Get())
	case *reflect.StructValue:
		t := v.Type().(*reflect.StructType)
		for i := 0; i < t.NumField(); i++ {
			var pre *forkableWriter
			pre, out = out.fork()
			err = marshalField(pre, v.Field(i), parseFieldParameters(t.Field(i).Tag))
			if err != nil {
				return
			}
		}
		return
	case *reflect.SliceValue:
		sliceType := v.Type().(*reflect.SliceType)
		if _, ok := sliceType.Elem().(*reflect.Uint8Type); ok {
			bytes := make([]byte, v.Len())
			for i := 0; i < v.Len(); i++ {
				bytes[i] = v.Elem(i).(*reflect.Uint8Value).Get()
			}
			_, err = out.Write(bytes)
			return
		}

		var params fieldParameters
		for i := 0; i < v.Len(); i++ {
			err = marshalField(out, v.Elem(i), params)
			if err != nil {
				return
			}
		}
		return
	case *reflect.StringValue:
		if params.stringType == tagIA5String {
			return marshalIA5String(out, v.Get())
		} else {
			return marshalPrintableString(out, v.Get())
		}
		return
	}

	return StructuralError{"unknown Go type"}
}

func marshalField(out *forkableWriter, v reflect.Value, params fieldParameters) (err os.Error) {
	tag, isCompound, ok := getUniversalType(v.Type())
	if !ok {
		err = StructuralError{fmt.Sprintf("unknown Go type: %v", v.Type())}
		return
	}
	class := classUniversal

	if params.stringType != 0 {
		if tag != tagPrintableString {
			return StructuralError{"Explicit string type given to non-string member"}
		}
		tag = params.stringType
	}

	tags, body := out.fork()

	err = marshalBody(body, v, params)
	if err != nil {
		return
	}

	bodyLen := body.Len()

	var explicitTag *forkableWriter
	if params.explicit {
		explicitTag, tags = tags.fork()
	}

	if !params.explicit && params.tag != nil {
		// implicit tag.
		tag = *params.tag
		class = classContextSpecific
	}

	err = marshalTagAndLength(tags, tagAndLength{class, tag, bodyLen, isCompound})
	if err != nil {
		return
	}

	if params.explicit {
		err = marshalTagAndLength(explicitTag, tagAndLength{
			class: classContextSpecific,
			tag: *params.tag,
			length: bodyLen + tags.Len(),
			isCompound: true,
		})
	}

	return nil
}

// Marshal serialises val as an ASN.1 structure and writes the result to out.
// In the case of an error, no output is produced.
func Marshal(out io.Writer, val interface{}) os.Error {
	v := reflect.NewValue(val)
	f := newForkableWriter()
	err := marshalField(f, v, fieldParameters{})
	if err != nil {
		return err
	}
	_, err = f.writeTo(out)
	return err
}
