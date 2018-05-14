// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptobyte

import (
	encoding_asn1 "encoding/asn1"
	"fmt"
	"math/big"
	"reflect"
	"time"

	"golang_org/x/crypto/cryptobyte/asn1"
)

// This file contains ASN.1-related methods for String and Builder.

// Builder

// AddASN1Int64 appends a DER-encoded ASN.1 INTEGER.
func (b *Builder) AddASN1Int64(v int64) {
	b.addASN1Signed(asn1.INTEGER, v)
}

// AddASN1Int64WithTag appends a DER-encoded ASN.1 INTEGER with the
// given tag.
func (b *Builder) AddASN1Int64WithTag(v int64, tag asn1.Tag) {
	b.addASN1Signed(tag, v)
}

// AddASN1Enum appends a DER-encoded ASN.1 ENUMERATION.
func (b *Builder) AddASN1Enum(v int64) {
	b.addASN1Signed(asn1.ENUM, v)
}

func (b *Builder) addASN1Signed(tag asn1.Tag, v int64) {
	b.AddASN1(tag, func(c *Builder) {
		length := 1
		for i := v; i >= 0x80 || i < -0x80; i >>= 8 {
			length++
		}

		for ; length > 0; length-- {
			i := v >> uint((length-1)*8) & 0xff
			c.AddUint8(uint8(i))
		}
	})
}

// AddASN1Uint64 appends a DER-encoded ASN.1 INTEGER.
func (b *Builder) AddASN1Uint64(v uint64) {
	b.AddASN1(asn1.INTEGER, func(c *Builder) {
		length := 1
		for i := v; i >= 0x80; i >>= 8 {
			length++
		}

		for ; length > 0; length-- {
			i := v >> uint((length-1)*8) & 0xff
			c.AddUint8(uint8(i))
		}
	})
}

// AddASN1BigInt appends a DER-encoded ASN.1 INTEGER.
func (b *Builder) AddASN1BigInt(n *big.Int) {
	if b.err != nil {
		return
	}

	b.AddASN1(asn1.INTEGER, func(c *Builder) {
		if n.Sign() < 0 {
			// A negative number has to be converted to two's-complement form. So we
			// invert and subtract 1. If the most-significant-bit isn't set then
			// we'll need to pad the beginning with 0xff in order to keep the number
			// negative.
			nMinus1 := new(big.Int).Neg(n)
			nMinus1.Sub(nMinus1, bigOne)
			bytes := nMinus1.Bytes()
			for i := range bytes {
				bytes[i] ^= 0xff
			}
			if bytes[0]&0x80 == 0 {
				c.add(0xff)
			}
			c.add(bytes...)
		} else if n.Sign() == 0 {
			c.add(0)
		} else {
			bytes := n.Bytes()
			if bytes[0]&0x80 != 0 {
				c.add(0)
			}
			c.add(bytes...)
		}
	})
}

// AddASN1OctetString appends a DER-encoded ASN.1 OCTET STRING.
func (b *Builder) AddASN1OctetString(bytes []byte) {
	b.AddASN1(asn1.OCTET_STRING, func(c *Builder) {
		c.AddBytes(bytes)
	})
}

const generalizedTimeFormatStr = "20060102150405Z0700"

// AddASN1GeneralizedTime appends a DER-encoded ASN.1 GENERALIZEDTIME.
func (b *Builder) AddASN1GeneralizedTime(t time.Time) {
	if t.Year() < 0 || t.Year() > 9999 {
		b.err = fmt.Errorf("cryptobyte: cannot represent %v as a GeneralizedTime", t)
		return
	}
	b.AddASN1(asn1.GeneralizedTime, func(c *Builder) {
		c.AddBytes([]byte(t.Format(generalizedTimeFormatStr)))
	})
}

// AddASN1BitString appends a DER-encoded ASN.1 BIT STRING. This does not
// support BIT STRINGs that are not a whole number of bytes.
func (b *Builder) AddASN1BitString(data []byte) {
	b.AddASN1(asn1.BIT_STRING, func(b *Builder) {
		b.AddUint8(0)
		b.AddBytes(data)
	})
}

func (b *Builder) addBase128Int(n int64) {
	var length int
	if n == 0 {
		length = 1
	} else {
		for i := n; i > 0; i >>= 7 {
			length++
		}
	}

	for i := length - 1; i >= 0; i-- {
		o := byte(n >> uint(i*7))
		o &= 0x7f
		if i != 0 {
			o |= 0x80
		}

		b.add(o)
	}
}

func isValidOID(oid encoding_asn1.ObjectIdentifier) bool {
	if len(oid) < 2 {
		return false
	}

	if oid[0] > 2 || (oid[0] <= 1 && oid[1] >= 40) {
		return false
	}

	for _, v := range oid {
		if v < 0 {
			return false
		}
	}

	return true
}

func (b *Builder) AddASN1ObjectIdentifier(oid encoding_asn1.ObjectIdentifier) {
	b.AddASN1(asn1.OBJECT_IDENTIFIER, func(b *Builder) {
		if !isValidOID(oid) {
			b.err = fmt.Errorf("cryptobyte: invalid OID: %v", oid)
			return
		}

		b.addBase128Int(int64(oid[0])*40 + int64(oid[1]))
		for _, v := range oid[2:] {
			b.addBase128Int(int64(v))
		}
	})
}

func (b *Builder) AddASN1Boolean(v bool) {
	b.AddASN1(asn1.BOOLEAN, func(b *Builder) {
		if v {
			b.AddUint8(0xff)
		} else {
			b.AddUint8(0)
		}
	})
}

func (b *Builder) AddASN1NULL() {
	b.add(uint8(asn1.NULL), 0)
}

// MarshalASN1 calls encoding_asn1.Marshal on its input and appends the result if
// successful or records an error if one occurred.
func (b *Builder) MarshalASN1(v interface{}) {
	// NOTE(martinkr): This is somewhat of a hack to allow propagation of
	// encoding_asn1.Marshal errors into Builder.err. N.B. if you call MarshalASN1 with a
	// value embedded into a struct, its tag information is lost.
	if b.err != nil {
		return
	}
	bytes, err := encoding_asn1.Marshal(v)
	if err != nil {
		b.err = err
		return
	}
	b.AddBytes(bytes)
}

// AddASN1 appends an ASN.1 object. The object is prefixed with the given tag.
// Tags greater than 30 are not supported and result in an error (i.e.
// low-tag-number form only). The child builder passed to the
// BuilderContinuation can be used to build the content of the ASN.1 object.
func (b *Builder) AddASN1(tag asn1.Tag, f BuilderContinuation) {
	if b.err != nil {
		return
	}
	// Identifiers with the low five bits set indicate high-tag-number format
	// (two or more octets), which we don't support.
	if tag&0x1f == 0x1f {
		b.err = fmt.Errorf("cryptobyte: high-tag number identifier octects not supported: 0x%x", tag)
		return
	}
	b.AddUint8(uint8(tag))
	b.addLengthPrefixed(1, true, f)
}

// String

// ReadASN1Boolean decodes an ASN.1 INTEGER and converts it to a boolean
// representation into out and advances. It reports whether the read
// was successful.
func (s *String) ReadASN1Boolean(out *bool) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.INTEGER) || len(bytes) != 1 {
		return false
	}

	switch bytes[0] {
	case 0:
		*out = false
	case 0xff:
		*out = true
	default:
		return false
	}

	return true
}

var bigIntType = reflect.TypeOf((*big.Int)(nil)).Elem()

// ReadASN1Integer decodes an ASN.1 INTEGER into out and advances. If out does
// not point to an integer or to a big.Int, it panics. It reports whether the
// read was successful.
func (s *String) ReadASN1Integer(out interface{}) bool {
	if reflect.TypeOf(out).Kind() != reflect.Ptr {
		panic("out is not a pointer")
	}
	switch reflect.ValueOf(out).Elem().Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		var i int64
		if !s.readASN1Int64(&i) || reflect.ValueOf(out).Elem().OverflowInt(i) {
			return false
		}
		reflect.ValueOf(out).Elem().SetInt(i)
		return true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		var u uint64
		if !s.readASN1Uint64(&u) || reflect.ValueOf(out).Elem().OverflowUint(u) {
			return false
		}
		reflect.ValueOf(out).Elem().SetUint(u)
		return true
	case reflect.Struct:
		if reflect.TypeOf(out).Elem() == bigIntType {
			return s.readASN1BigInt(out.(*big.Int))
		}
	}
	panic("out does not point to an integer type")
}

func checkASN1Integer(bytes []byte) bool {
	if len(bytes) == 0 {
		// An INTEGER is encoded with at least one octet.
		return false
	}
	if len(bytes) == 1 {
		return true
	}
	if bytes[0] == 0 && bytes[1]&0x80 == 0 || bytes[0] == 0xff && bytes[1]&0x80 == 0x80 {
		// Value is not minimally encoded.
		return false
	}
	return true
}

var bigOne = big.NewInt(1)

func (s *String) readASN1BigInt(out *big.Int) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.INTEGER) || !checkASN1Integer(bytes) {
		return false
	}
	if bytes[0]&0x80 == 0x80 {
		// Negative number.
		neg := make([]byte, len(bytes))
		for i, b := range bytes {
			neg[i] = ^b
		}
		out.SetBytes(neg)
		out.Add(out, bigOne)
		out.Neg(out)
	} else {
		out.SetBytes(bytes)
	}
	return true
}

func (s *String) readASN1Int64(out *int64) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.INTEGER) || !checkASN1Integer(bytes) || !asn1Signed(out, bytes) {
		return false
	}
	return true
}

func asn1Signed(out *int64, n []byte) bool {
	length := len(n)
	if length > 8 {
		return false
	}
	for i := 0; i < length; i++ {
		*out <<= 8
		*out |= int64(n[i])
	}
	// Shift up and down in order to sign extend the result.
	*out <<= 64 - uint8(length)*8
	*out >>= 64 - uint8(length)*8
	return true
}

func (s *String) readASN1Uint64(out *uint64) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.INTEGER) || !checkASN1Integer(bytes) || !asn1Unsigned(out, bytes) {
		return false
	}
	return true
}

func asn1Unsigned(out *uint64, n []byte) bool {
	length := len(n)
	if length > 9 || length == 9 && n[0] != 0 {
		// Too large for uint64.
		return false
	}
	if n[0]&0x80 != 0 {
		// Negative number.
		return false
	}
	for i := 0; i < length; i++ {
		*out <<= 8
		*out |= uint64(n[i])
	}
	return true
}

// ReadASN1Int64WithTag decodes an ASN.1 INTEGER with the given tag into out
// and advances. It reports whether the read was successful and resulted in a
// value that can be represented in an int64.
func (s *String) ReadASN1Int64WithTag(out *int64, tag asn1.Tag) bool {
	var bytes String
	return s.ReadASN1(&bytes, tag) && checkASN1Integer(bytes) && asn1Signed(out, bytes)
}

// ReadASN1Enum decodes an ASN.1 ENUMERATION into out and advances. It reports
// whether the read was successful.
func (s *String) ReadASN1Enum(out *int) bool {
	var bytes String
	var i int64
	if !s.ReadASN1(&bytes, asn1.ENUM) || !checkASN1Integer(bytes) || !asn1Signed(&i, bytes) {
		return false
	}
	if int64(int(i)) != i {
		return false
	}
	*out = int(i)
	return true
}

func (s *String) readBase128Int(out *int) bool {
	ret := 0
	for i := 0; len(*s) > 0; i++ {
		if i == 4 {
			return false
		}
		ret <<= 7
		b := s.read(1)[0]
		ret |= int(b & 0x7f)
		if b&0x80 == 0 {
			*out = ret
			return true
		}
	}
	return false // truncated
}

// ReadASN1ObjectIdentifier decodes an ASN.1 OBJECT IDENTIFIER into out and
// advances. It reports whether the read was successful.
func (s *String) ReadASN1ObjectIdentifier(out *encoding_asn1.ObjectIdentifier) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.OBJECT_IDENTIFIER) || len(bytes) == 0 {
		return false
	}

	// In the worst case, we get two elements from the first byte (which is
	// encoded differently) and then every varint is a single byte long.
	components := make([]int, len(bytes)+1)

	// The first varint is 40*value1 + value2:
	// According to this packing, value1 can take the values 0, 1 and 2 only.
	// When value1 = 0 or value1 = 1, then value2 is <= 39. When value1 = 2,
	// then there are no restrictions on value2.
	var v int
	if !bytes.readBase128Int(&v) {
		return false
	}
	if v < 80 {
		components[0] = v / 40
		components[1] = v % 40
	} else {
		components[0] = 2
		components[1] = v - 80
	}

	i := 2
	for ; len(bytes) > 0; i++ {
		if !bytes.readBase128Int(&v) {
			return false
		}
		components[i] = v
	}
	*out = components[:i]
	return true
}

// ReadASN1GeneralizedTime decodes an ASN.1 GENERALIZEDTIME into out and
// advances. It reports whether the read was successful.
func (s *String) ReadASN1GeneralizedTime(out *time.Time) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.GeneralizedTime) {
		return false
	}
	t := string(bytes)
	res, err := time.Parse(generalizedTimeFormatStr, t)
	if err != nil {
		return false
	}
	if serialized := res.Format(generalizedTimeFormatStr); serialized != t {
		return false
	}
	*out = res
	return true
}

// ReadASN1BitString decodes an ASN.1 BIT STRING into out and advances.
// It reports whether the read was successful.
func (s *String) ReadASN1BitString(out *encoding_asn1.BitString) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.BIT_STRING) || len(bytes) == 0 {
		return false
	}

	paddingBits := uint8(bytes[0])
	bytes = bytes[1:]
	if paddingBits > 7 ||
		len(bytes) == 0 && paddingBits != 0 ||
		len(bytes) > 0 && bytes[len(bytes)-1]&(1<<paddingBits-1) != 0 {
		return false
	}

	out.BitLength = len(bytes)*8 - int(paddingBits)
	out.Bytes = bytes
	return true
}

// ReadASN1BitString decodes an ASN.1 BIT STRING into out and advances. It is
// an error if the BIT STRING is not a whole number of bytes. It reports
// whether the read was successful.
func (s *String) ReadASN1BitStringAsBytes(out *[]byte) bool {
	var bytes String
	if !s.ReadASN1(&bytes, asn1.BIT_STRING) || len(bytes) == 0 {
		return false
	}

	paddingBits := uint8(bytes[0])
	if paddingBits != 0 {
		return false
	}
	*out = bytes[1:]
	return true
}

// ReadASN1Bytes reads the contents of a DER-encoded ASN.1 element (not including
// tag and length bytes) into out, and advances. The element must match the
// given tag. It reports whether the read was successful.
func (s *String) ReadASN1Bytes(out *[]byte, tag asn1.Tag) bool {
	return s.ReadASN1((*String)(out), tag)
}

// ReadASN1 reads the contents of a DER-encoded ASN.1 element (not including
// tag and length bytes) into out, and advances. The element must match the
// given tag. It reports whether the read was successful.
//
// Tags greater than 30 are not supported (i.e. low-tag-number format only).
func (s *String) ReadASN1(out *String, tag asn1.Tag) bool {
	var t asn1.Tag
	if !s.ReadAnyASN1(out, &t) || t != tag {
		return false
	}
	return true
}

// ReadASN1Element reads the contents of a DER-encoded ASN.1 element (including
// tag and length bytes) into out, and advances. The element must match the
// given tag. It reports whether the read was successful.
//
// Tags greater than 30 are not supported (i.e. low-tag-number format only).
func (s *String) ReadASN1Element(out *String, tag asn1.Tag) bool {
	var t asn1.Tag
	if !s.ReadAnyASN1Element(out, &t) || t != tag {
		return false
	}
	return true
}

// ReadAnyASN1 reads the contents of a DER-encoded ASN.1 element (not including
// tag and length bytes) into out, sets outTag to its tag, and advances.
// It reports whether the read was successful.
//
// Tags greater than 30 are not supported (i.e. low-tag-number format only).
func (s *String) ReadAnyASN1(out *String, outTag *asn1.Tag) bool {
	return s.readASN1(out, outTag, true /* skip header */)
}

// ReadAnyASN1Element reads the contents of a DER-encoded ASN.1 element
// (including tag and length bytes) into out, sets outTag to is tag, and
// advances. It reports whether the read was successful.
//
// Tags greater than 30 are not supported (i.e. low-tag-number format only).
func (s *String) ReadAnyASN1Element(out *String, outTag *asn1.Tag) bool {
	return s.readASN1(out, outTag, false /* include header */)
}

// PeekASN1Tag reports whether the next ASN.1 value on the string starts with
// the given tag.
func (s String) PeekASN1Tag(tag asn1.Tag) bool {
	if len(s) == 0 {
		return false
	}
	return asn1.Tag(s[0]) == tag
}

// SkipASN1 reads and discards an ASN.1 element with the given tag. It
// reports whether the operation was successful.
func (s *String) SkipASN1(tag asn1.Tag) bool {
	var unused String
	return s.ReadASN1(&unused, tag)
}

// ReadOptionalASN1 attempts to read the contents of a DER-encoded ASN.1
// element (not including tag and length bytes) tagged with the given tag into
// out. It stores whether an element with the tag was found in outPresent,
// unless outPresent is nil. It reports whether the read was successful.
func (s *String) ReadOptionalASN1(out *String, outPresent *bool, tag asn1.Tag) bool {
	present := s.PeekASN1Tag(tag)
	if outPresent != nil {
		*outPresent = present
	}
	if present && !s.ReadASN1(out, tag) {
		return false
	}
	return true
}

// SkipOptionalASN1 advances s over an ASN.1 element with the given tag, or
// else leaves s unchanged. It reports whether the operation was successful.
func (s *String) SkipOptionalASN1(tag asn1.Tag) bool {
	if !s.PeekASN1Tag(tag) {
		return true
	}
	var unused String
	return s.ReadASN1(&unused, tag)
}

// ReadOptionalASN1Integer attempts to read an optional ASN.1 INTEGER
// explicitly tagged with tag into out and advances. If no element with a
// matching tag is present, it writes defaultValue into out instead. If out
// does not point to an integer or to a big.Int, it panics. It reports
// whether the read was successful.
func (s *String) ReadOptionalASN1Integer(out interface{}, tag asn1.Tag, defaultValue interface{}) bool {
	if reflect.TypeOf(out).Kind() != reflect.Ptr {
		panic("out is not a pointer")
	}
	var present bool
	var i String
	if !s.ReadOptionalASN1(&i, &present, tag) {
		return false
	}
	if !present {
		switch reflect.ValueOf(out).Elem().Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			reflect.ValueOf(out).Elem().Set(reflect.ValueOf(defaultValue))
		case reflect.Struct:
			if reflect.TypeOf(out).Elem() != bigIntType {
				panic("invalid integer type")
			}
			if reflect.TypeOf(defaultValue).Kind() != reflect.Ptr ||
				reflect.TypeOf(defaultValue).Elem() != bigIntType {
				panic("out points to big.Int, but defaultValue does not")
			}
			out.(*big.Int).Set(defaultValue.(*big.Int))
		default:
			panic("invalid integer type")
		}
		return true
	}
	if !i.ReadASN1Integer(out) || !i.Empty() {
		return false
	}
	return true
}

// ReadOptionalASN1OctetString attempts to read an optional ASN.1 OCTET STRING
// explicitly tagged with tag into out and advances. If no element with a
// matching tag is present, it sets "out" to nil instead. It reports
// whether the read was successful.
func (s *String) ReadOptionalASN1OctetString(out *[]byte, outPresent *bool, tag asn1.Tag) bool {
	var present bool
	var child String
	if !s.ReadOptionalASN1(&child, &present, tag) {
		return false
	}
	if outPresent != nil {
		*outPresent = present
	}
	if present {
		var oct String
		if !child.ReadASN1(&oct, asn1.OCTET_STRING) || !child.Empty() {
			return false
		}
		*out = oct
	} else {
		*out = nil
	}
	return true
}

// ReadOptionalASN1Boolean sets *out to the value of the next ASN.1 BOOLEAN or,
// if the next bytes are not an ASN.1 BOOLEAN, to the value of defaultValue.
// It reports whether the operation was successful.
func (s *String) ReadOptionalASN1Boolean(out *bool, defaultValue bool) bool {
	var present bool
	var child String
	if !s.ReadOptionalASN1(&child, &present, asn1.BOOLEAN) {
		return false
	}

	if !present {
		*out = defaultValue
		return true
	}

	return s.ReadASN1Boolean(out)
}

func (s *String) readASN1(out *String, outTag *asn1.Tag, skipHeader bool) bool {
	if len(*s) < 2 {
		return false
	}
	tag, lenByte := (*s)[0], (*s)[1]

	if tag&0x1f == 0x1f {
		// ITU-T X.690 section 8.1.2
		//
		// An identifier octet with a tag part of 0x1f indicates a high-tag-number
		// form identifier with two or more octets. We only support tags less than
		// 31 (i.e. low-tag-number form, single octet identifier).
		return false
	}

	if outTag != nil {
		*outTag = asn1.Tag(tag)
	}

	// ITU-T X.690 section 8.1.3
	//
	// Bit 8 of the first length byte indicates whether the length is short- or
	// long-form.
	var length, headerLen uint32 // length includes headerLen
	if lenByte&0x80 == 0 {
		// Short-form length (section 8.1.3.4), encoded in bits 1-7.
		length = uint32(lenByte) + 2
		headerLen = 2
	} else {
		// Long-form length (section 8.1.3.5). Bits 1-7 encode the number of octets
		// used to encode the length.
		lenLen := lenByte & 0x7f
		var len32 uint32

		if lenLen == 0 || lenLen > 4 || len(*s) < int(2+lenLen) {
			return false
		}

		lenBytes := String((*s)[2 : 2+lenLen])
		if !lenBytes.readUnsigned(&len32, int(lenLen)) {
			return false
		}

		// ITU-T X.690 section 10.1 (DER length forms) requires encoding the length
		// with the minimum number of octets.
		if len32 < 128 {
			// Length should have used short-form encoding.
			return false
		}
		if len32>>((lenLen-1)*8) == 0 {
			// Leading octet is 0. Length should have been at least one byte shorter.
			return false
		}

		headerLen = 2 + uint32(lenLen)
		if headerLen+len32 < len32 {
			// Overflow.
			return false
		}
		length = headerLen + len32
	}

	if uint32(int(length)) != length || !s.ReadBytes((*[]byte)(out), int(length)) {
		return false
	}
	if skipHeader && !out.Skip(int(headerLen)) {
		panic("cryptobyte: internal error")
	}

	return true
}
