// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asn1

import (
	"bytes"
	"errors"
	"fmt"
	"math/big"
	"reflect"
	"sort"
	"time"
	"unicode/utf8"
)

var (
	byte00Encoder encoder = byteEncoder(0x00)
	byteFFEncoder encoder = byteEncoder(0xff)
)

// encoder represents an ASN.1 element that is waiting to be marshaled.
type encoder interface {
	// Len returns the number of bytes needed to marshal this element.
	Len() int
	// Encode encodes this element by writing Len() bytes to dst.
	Encode(dst []byte)
}

type byteEncoder byte

func (c byteEncoder) Len() int {
	return 1
}

func (c byteEncoder) Encode(dst []byte) {
	dst[0] = byte(c)
}

type bytesEncoder []byte

func (b bytesEncoder) Len() int {
	return len(b)
}

func (b bytesEncoder) Encode(dst []byte) {
	if copy(dst, b) != len(b) {
		panic("internal error")
	}
}

type stringEncoder string

func (s stringEncoder) Len() int {
	return len(s)
}

func (s stringEncoder) Encode(dst []byte) {
	if copy(dst, s) != len(s) {
		panic("internal error")
	}
}

type multiEncoder []encoder

func (m multiEncoder) Len() int {
	var size int
	for _, e := range m {
		size += e.Len()
	}
	return size
}

func (m multiEncoder) Encode(dst []byte) {
	var off int
	for _, e := range m {
		e.Encode(dst[off:])
		off += e.Len()
	}
}

type setEncoder []encoder

func (s setEncoder) Len() int {
	var size int
	for _, e := range s {
		size += e.Len()
	}
	return size
}

func (s setEncoder) Encode(dst []byte) {
	// Per X690 Section 11.6: The encodings of the component values of a
	// set-of value shall appear in ascending order, the encodings being
	// compared as octet strings with the shorter components being padded
	// at their trailing end with 0-octets.
	//
	// First we encode each element to its TLV encoding and then use
	// octetSort to get the ordering expected by X690 DER rules before
	// writing the sorted encodings out to dst.
	l := make([][]byte, len(s))
	for i, e := range s {
		l[i] = make([]byte, e.Len())
		e.Encode(l[i])
	}

	sort.Slice(l, func(i, j int) bool {
		// Since we are using bytes.Compare to compare TLV encodings we
		// don't need to right pad s[i] and s[j] to the same length as
		// suggested in X690. If len(s[i]) < len(s[j]) the length octet of
		// s[i], which is the first determining byte, will inherently be
		// smaller than the length octet of s[j]. This lets us skip the
		// padding step.
		return bytes.Compare(l[i], l[j]) < 0
	})

	var off int
	for _, b := range l {
		copy(dst[off:], b)
		off += len(b)
	}
}

type taggedEncoder struct {
	// scratch contains temporary space for encoding the tag and length of
	// an element in order to avoid extra allocations.
	scratch [8]byte
	tag     encoder
	body    encoder
}

func (t *taggedEncoder) Len() int {
	return t.tag.Len() + t.body.Len()
}

func (t *taggedEncoder) Encode(dst []byte) {
	t.tag.Encode(dst)
	t.body.Encode(dst[t.tag.Len():])
}

type int64Encoder int64

func (i int64Encoder) Len() int {
	n := 1

	for i > 127 {
		n++
		i >>= 8
	}

	for i < -128 {
		n++
		i >>= 8
	}

	return n
}

func (i int64Encoder) Encode(dst []byte) {
	n := i.Len()

	for j := 0; j < n; j++ {
		dst[j] = byte(i >> uint((n-1-j)*8))
	}
}

func base128IntLength(n int64) int {
	if n == 0 {
		return 1
	}

	l := 0
	for i := n; i > 0; i >>= 7 {
		l++
	}

	return l
}

func appendBase128Int(dst []byte, n int64) []byte {
	l := base128IntLength(n)

	for i := l - 1; i >= 0; i-- {
		o := byte(n >> uint(i*7))
		o &= 0x7f
		if i != 0 {
			o |= 0x80
		}

		dst = append(dst, o)
	}

	return dst
}

func makeBigInt(n *big.Int) (encoder, error) {
	if n == nil {
		return nil, StructuralError{"empty integer"}
	}

	if n.Sign() < 0 {
		// A negative number has to be converted to two's-complement
		// form. So we'll invert and subtract 1. If the
		// most-significant-bit isn't set then we'll need to pad the
		// beginning with 0xff in order to keep the number negative.
		nMinus1 := new(big.Int).Neg(n)
		nMinus1.Sub(nMinus1, bigOne)
		bytes := nMinus1.Bytes()
		for i := range bytes {
			bytes[i] ^= 0xff
		}
		if len(bytes) == 0 || bytes[0]&0x80 == 0 {
			return multiEncoder([]encoder{byteFFEncoder, bytesEncoder(bytes)}), nil
		}
		return bytesEncoder(bytes), nil
	} else if n.Sign() == 0 {
		// Zero is written as a single 0 zero rather than no bytes.
		return byte00Encoder, nil
	} else {
		bytes := n.Bytes()
		if len(bytes) > 0 && bytes[0]&0x80 != 0 {
			// We'll have to pad this with 0x00 in order to stop it
			// looking like a negative number.
			return multiEncoder([]encoder{byte00Encoder, bytesEncoder(bytes)}), nil
		}
		return bytesEncoder(bytes), nil
	}
}

func appendLength(dst []byte, i int) []byte {
	n := lengthLength(i)

	for ; n > 0; n-- {
		dst = append(dst, byte(i>>uint((n-1)*8)))
	}

	return dst
}

func lengthLength(i int) (numBytes int) {
	numBytes = 1
	for i > 255 {
		numBytes++
		i >>= 8
	}
	return
}

func appendTagAndLength(dst []byte, t tagAndLength) []byte {
	b := uint8(t.class) << 6
	if t.isCompound {
		b |= 0x20
	}
	if t.tag >= 31 {
		b |= 0x1f
		dst = append(dst, b)
		dst = appendBase128Int(dst, int64(t.tag))
	} else {
		b |= uint8(t.tag)
		dst = append(dst, b)
	}

	if t.length >= 128 {
		l := lengthLength(t.length)
		dst = append(dst, 0x80|byte(l))
		dst = appendLength(dst, t.length)
	} else {
		dst = append(dst, byte(t.length))
	}

	return dst
}

type bitStringEncoder BitString

func (b bitStringEncoder) Len() int {
	return len(b.Bytes) + 1
}

func (b bitStringEncoder) Encode(dst []byte) {
	dst[0] = byte((8 - b.BitLength%8) % 8)
	if copy(dst[1:], b.Bytes) != len(b.Bytes) {
		panic("internal error")
	}
}

type oidEncoder []int

func (oid oidEncoder) Len() int {
	l := base128IntLength(int64(oid[0]*40 + oid[1]))
	for i := 2; i < len(oid); i++ {
		l += base128IntLength(int64(oid[i]))
	}
	return l
}

func (oid oidEncoder) Encode(dst []byte) {
	dst = appendBase128Int(dst[:0], int64(oid[0]*40+oid[1]))
	for i := 2; i < len(oid); i++ {
		dst = appendBase128Int(dst, int64(oid[i]))
	}
}

func makeObjectIdentifier(oid []int) (e encoder, err error) {
	if len(oid) < 2 || oid[0] > 2 || (oid[0] < 2 && oid[1] >= 40) {
		return nil, StructuralError{"invalid object identifier"}
	}

	return oidEncoder(oid), nil
}

func makePrintableString(s string) (e encoder, err error) {
	for i := 0; i < len(s); i++ {
		// The asterisk is often used in PrintableString, even though
		// it is invalid. If a PrintableString was specifically
		// requested then the asterisk is permitted by this code.
		// Ampersand is allowed in parsing due a handful of CA
		// certificates, however when making new certificates
		// it is rejected.
		if !isPrintable(s[i], allowAsterisk, rejectAmpersand) {
			return nil, StructuralError{"PrintableString contains invalid character"}
		}
	}

	return stringEncoder(s), nil
}

func makeIA5String(s string) (e encoder, err error) {
	for i := 0; i < len(s); i++ {
		if s[i] > 127 {
			return nil, StructuralError{"IA5String contains invalid character"}
		}
	}

	return stringEncoder(s), nil
}

func makeNumericString(s string) (e encoder, err error) {
	for i := 0; i < len(s); i++ {
		if !isNumeric(s[i]) {
			return nil, StructuralError{"NumericString contains invalid character"}
		}
	}

	return stringEncoder(s), nil
}

func makeUTF8String(s string) encoder {
	return stringEncoder(s)
}

func appendTwoDigits(dst []byte, v int) []byte {
	return append(dst, byte('0'+(v/10)%10), byte('0'+v%10))
}

func appendFourDigits(dst []byte, v int) []byte {
	var bytes [4]byte
	for i := range bytes {
		bytes[3-i] = '0' + byte(v%10)
		v /= 10
	}
	return append(dst, bytes[:]...)
}

func outsideUTCRange(t time.Time) bool {
	year := t.Year()
	return year < 1950 || year >= 2050
}

func makeUTCTime(t time.Time) (e encoder, err error) {
	dst := make([]byte, 0, 18)

	dst, err = appendUTCTime(dst, t)
	if err != nil {
		return nil, err
	}

	return bytesEncoder(dst), nil
}

func makeGeneralizedTime(t time.Time) (e encoder, err error) {
	dst := make([]byte, 0, 20)

	dst, err = appendGeneralizedTime(dst, t)
	if err != nil {
		return nil, err
	}

	return bytesEncoder(dst), nil
}

func appendUTCTime(dst []byte, t time.Time) (ret []byte, err error) {
	year := t.Year()

	switch {
	case 1950 <= year && year < 2000:
		dst = appendTwoDigits(dst, year-1900)
	case 2000 <= year && year < 2050:
		dst = appendTwoDigits(dst, year-2000)
	default:
		return nil, StructuralError{"cannot represent time as UTCTime"}
	}

	return appendTimeCommon(dst, t), nil
}

func appendGeneralizedTime(dst []byte, t time.Time) (ret []byte, err error) {
	year := t.Year()
	if year < 0 || year > 9999 {
		return nil, StructuralError{"cannot represent time as GeneralizedTime"}
	}

	dst = appendFourDigits(dst, year)

	return appendTimeCommon(dst, t), nil
}

func appendTimeCommon(dst []byte, t time.Time) []byte {
	_, month, day := t.Date()

	dst = appendTwoDigits(dst, int(month))
	dst = appendTwoDigits(dst, day)

	hour, min, sec := t.Clock()

	dst = appendTwoDigits(dst, hour)
	dst = appendTwoDigits(dst, min)
	dst = appendTwoDigits(dst, sec)

	_, offset := t.Zone()

	switch {
	case offset/60 == 0:
		return append(dst, 'Z')
	case offset > 0:
		dst = append(dst, '+')
	case offset < 0:
		dst = append(dst, '-')
	}

	offsetMinutes := offset / 60
	if offsetMinutes < 0 {
		offsetMinutes = -offsetMinutes
	}

	dst = appendTwoDigits(dst, offsetMinutes/60)
	dst = appendTwoDigits(dst, offsetMinutes%60)

	return dst
}

func stripTagAndLength(in []byte) []byte {
	_, offset, err := parseTagAndLength(in, 0)
	if err != nil {
		return in
	}
	return in[offset:]
}

func makeBody(value reflect.Value, params fieldParameters) (e encoder, err error) {
	switch value.Type() {
	case flagType:
		return bytesEncoder(nil), nil
	case timeType:
		t := value.Interface().(time.Time)
		if params.timeType == TagGeneralizedTime || outsideUTCRange(t) {
			return makeGeneralizedTime(t)
		}
		return makeUTCTime(t)
	case bitStringType:
		return bitStringEncoder(value.Interface().(BitString)), nil
	case objectIdentifierType:
		return makeObjectIdentifier(value.Interface().(ObjectIdentifier))
	case bigIntType:
		return makeBigInt(value.Interface().(*big.Int))
	}

	switch v := value; v.Kind() {
	case reflect.Bool:
		if v.Bool() {
			return byteFFEncoder, nil
		}
		return byte00Encoder, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int64Encoder(v.Int()), nil
	case reflect.Struct:
		t := v.Type()

		for i := 0; i < t.NumField(); i++ {
			if !t.Field(i).IsExported() {
				return nil, StructuralError{"struct contains unexported fields"}
			}
		}

		startingField := 0

		n := t.NumField()
		if n == 0 {
			return bytesEncoder(nil), nil
		}

		// If the first element of the structure is a non-empty
		// RawContents, then we don't bother serializing the rest.
		if t.Field(0).Type == rawContentsType {
			s := v.Field(0)
			if s.Len() > 0 {
				bytes := s.Bytes()
				/* The RawContents will contain the tag and
				 * length fields but we'll also be writing
				 * those ourselves, so we strip them out of
				 * bytes */
				return bytesEncoder(stripTagAndLength(bytes)), nil
			}

			startingField = 1
		}

		switch n1 := n - startingField; n1 {
		case 0:
			return bytesEncoder(nil), nil
		case 1:
			return makeField(v.Field(startingField), parseFieldParameters(t.Field(startingField).Tag.Get("asn1")))
		default:
			m := make([]encoder, n1)
			for i := 0; i < n1; i++ {
				m[i], err = makeField(v.Field(i+startingField), parseFieldParameters(t.Field(i+startingField).Tag.Get("asn1")))
				if err != nil {
					return nil, err
				}
			}

			return multiEncoder(m), nil
		}
	case reflect.Slice:
		sliceType := v.Type()
		if sliceType.Elem().Kind() == reflect.Uint8 {
			return bytesEncoder(v.Bytes()), nil
		}

		var fp fieldParameters

		switch l := v.Len(); l {
		case 0:
			return bytesEncoder(nil), nil
		case 1:
			return makeField(v.Index(0), fp)
		default:
			m := make([]encoder, l)

			for i := 0; i < l; i++ {
				m[i], err = makeField(v.Index(i), fp)
				if err != nil {
					return nil, err
				}
			}

			if params.set {
				return setEncoder(m), nil
			}
			return multiEncoder(m), nil
		}
	case reflect.String:
		switch params.stringType {
		case TagIA5String:
			return makeIA5String(v.String())
		case TagPrintableString:
			return makePrintableString(v.String())
		case TagNumericString:
			return makeNumericString(v.String())
		default:
			return makeUTF8String(v.String()), nil
		}
	}

	return nil, StructuralError{"unknown Go type"}
}

func makeField(v reflect.Value, params fieldParameters) (e encoder, err error) {
	if !v.IsValid() {
		return nil, errors.New("asn1: cannot marshal nil value")
	}
	// If the field is an interface{} then recurse into it.
	if v.Kind() == reflect.Interface && v.Type().NumMethod() == 0 {
		return makeField(v.Elem(), params)
	}

	if v.Kind() == reflect.Slice && v.Len() == 0 && params.omitEmpty {
		return bytesEncoder(nil), nil
	}

	if params.optional && params.defaultValue != nil && canHaveDefaultValue(v.Kind()) {
		defaultValue := reflect.New(v.Type()).Elem()
		defaultValue.SetInt(*params.defaultValue)

		if reflect.DeepEqual(v.Interface(), defaultValue.Interface()) {
			return bytesEncoder(nil), nil
		}
	}

	// If no default value is given then the zero value for the type is
	// assumed to be the default value. This isn't obviously the correct
	// behavior, but it's what Go has traditionally done.
	if params.optional && params.defaultValue == nil {
		if reflect.DeepEqual(v.Interface(), reflect.Zero(v.Type()).Interface()) {
			return bytesEncoder(nil), nil
		}
	}

	if v.Type() == rawValueType {
		rv := v.Interface().(RawValue)
		if len(rv.FullBytes) != 0 {
			return bytesEncoder(rv.FullBytes), nil
		}

		t := new(taggedEncoder)

		t.tag = bytesEncoder(appendTagAndLength(t.scratch[:0], tagAndLength{rv.Class, rv.Tag, len(rv.Bytes), rv.IsCompound}))
		t.body = bytesEncoder(rv.Bytes)

		return t, nil
	}

	matchAny, tag, isCompound, ok := getUniversalType(v.Type())
	if !ok || matchAny {
		return nil, StructuralError{fmt.Sprintf("unknown Go type: %v", v.Type())}
	}

	if params.timeType != 0 && tag != TagUTCTime {
		return nil, StructuralError{"explicit time type given to non-time member"}
	}

	if params.stringType != 0 && tag != TagPrintableString {
		return nil, StructuralError{"explicit string type given to non-string member"}
	}

	switch tag {
	case TagPrintableString:
		if params.stringType == 0 {
			// This is a string without an explicit string type. We'll use
			// a PrintableString if the character set in the string is
			// sufficiently limited, otherwise we'll use a UTF8String.
			for _, r := range v.String() {
				if r >= utf8.RuneSelf || !isPrintable(byte(r), rejectAsterisk, rejectAmpersand) {
					if !utf8.ValidString(v.String()) {
						return nil, errors.New("asn1: string not valid UTF-8")
					}
					tag = TagUTF8String
					break
				}
			}
		} else {
			tag = params.stringType
		}
	case TagUTCTime:
		if params.timeType == TagGeneralizedTime || outsideUTCRange(v.Interface().(time.Time)) {
			tag = TagGeneralizedTime
		}
	}

	if params.set {
		if tag != TagSequence {
			return nil, StructuralError{"non sequence tagged as set"}
		}
		tag = TagSet
	}

	// makeField can be called for a slice that should be treated as a SET
	// but doesn't have params.set set, for instance when using a slice
	// with the SET type name suffix. In this case getUniversalType returns
	// TagSet, but makeBody doesn't know about that so will treat the slice
	// as a sequence. To work around this we set params.set.
	if tag == TagSet && !params.set {
		params.set = true
	}

	t := new(taggedEncoder)

	t.body, err = makeBody(v, params)
	if err != nil {
		return nil, err
	}

	bodyLen := t.body.Len()

	class := ClassUniversal
	if params.tag != nil {
		if params.application {
			class = ClassApplication
		} else if params.private {
			class = ClassPrivate
		} else {
			class = ClassContextSpecific
		}

		if params.explicit {
			t.tag = bytesEncoder(appendTagAndLength(t.scratch[:0], tagAndLength{ClassUniversal, tag, bodyLen, isCompound}))

			tt := new(taggedEncoder)

			tt.body = t

			tt.tag = bytesEncoder(appendTagAndLength(tt.scratch[:0], tagAndLength{
				class:      class,
				tag:        *params.tag,
				length:     bodyLen + t.tag.Len(),
				isCompound: true,
			}))

			return tt, nil
		}

		// implicit tag.
		tag = *params.tag
	}

	t.tag = bytesEncoder(appendTagAndLength(t.scratch[:0], tagAndLength{class, tag, bodyLen, isCompound}))

	return t, nil
}

// Marshal returns the ASN.1 encoding of val.
//
// In addition to the struct tags recognized by Unmarshal, the following can be
// used:
//
//	ia5:         causes strings to be marshaled as ASN.1, IA5String values
//	omitempty:   causes empty slices to be skipped
//	printable:   causes strings to be marshaled as ASN.1, PrintableString values
//	utf8:        causes strings to be marshaled as ASN.1, UTF8String values
//	utc:         causes time.Time to be marshaled as ASN.1, UTCTime values
//	generalized: causes time.Time to be marshaled as ASN.1, GeneralizedTime values
func Marshal(val any) ([]byte, error) {
	return MarshalWithParams(val, "")
}

// MarshalWithParams allows field parameters to be specified for the
// top-level element. The form of the params is the same as the field tags.
func MarshalWithParams(val any, params string) ([]byte, error) {
	e, err := makeField(reflect.ValueOf(val), parseFieldParameters(params))
	if err != nil {
		return nil, err
	}
	b := make([]byte, e.Len())
	e.Encode(b)
	return b, nil
}
