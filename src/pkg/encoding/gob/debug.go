// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Delete the next line to include in the gob package.
// +build ignore

package gob

// This file is not normally included in the gob package.  Used only for debugging the package itself.
// Except for reading uints, it is an implementation of a reader that is independent of
// the one implemented by Decoder.
// To enable the Debug function, delete the +build ignore line above and do
//	go install

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

var dumpBytes = false // If true, print the remaining bytes in the input buffer at each item.

// Init installs the debugging facility. If this file is not compiled in the
// package, the tests in codec_test.go are no-ops.
func init() {
	debugFunc = Debug
}

var (
	blanks = bytes.Repeat([]byte{' '}, 3*10)
	empty  = []byte(": <empty>\n")
	tabs   = strings.Repeat("\t", 100)
)

// tab indents itself when printed.
type tab int

func (t tab) String() string {
	n := int(t)
	if n > len(tabs) {
		n = len(tabs)
	}
	return tabs[0:n]
}

func (t tab) print() {
	fmt.Fprint(os.Stderr, t)
}

// A peekReader wraps an io.Reader, allowing one to peek ahead to see
// what's coming without stealing the data from the client of the Reader.
type peekReader struct {
	r    io.Reader
	data []byte // read-ahead data
}

// newPeekReader returns a peekReader that wraps r.
func newPeekReader(r io.Reader) *peekReader {
	return &peekReader{r: r}
}

// Read is the usual method. It will first take data that has been read ahead.
func (p *peekReader) Read(b []byte) (n int, err error) {
	if len(p.data) == 0 {
		return p.r.Read(b)
	}
	// Satisfy what's possible from the read-ahead data.
	n = copy(b, p.data)
	// Move data down to beginning of slice, to avoid endless growth
	copy(p.data, p.data[n:])
	p.data = p.data[:len(p.data)-n]
	return
}

// peek returns as many bytes as possible from the unread
// portion of the stream, up to the length of b.
func (p *peekReader) peek(b []byte) (n int, err error) {
	if len(p.data) > 0 {
		n = copy(b, p.data)
		if n == len(b) {
			return
		}
		b = b[n:]
	}
	if len(b) == 0 {
		return
	}
	m, e := io.ReadFull(p.r, b)
	if m > 0 {
		p.data = append(p.data, b[:m]...)
	}
	n += m
	if e == io.ErrUnexpectedEOF {
		// That means m > 0 but we reached EOF. If we got data
		// we won't complain about not being able to peek enough.
		if n > 0 {
			e = nil
		} else {
			e = io.EOF
		}
	}
	return n, e
}

type debugger struct {
	mutex          sync.Mutex
	remain         int  // the number of bytes known to remain in the input
	remainingKnown bool // the value of 'remain' is valid
	r              *peekReader
	wireType       map[typeId]*wireType
	tmp            []byte // scratch space for decoding uints.
}

// dump prints the next nBytes of the input.
// It arranges to print the output aligned from call to
// call, to make it easy to see what has been consumed.
func (deb *debugger) dump(format string, args ...interface{}) {
	if !dumpBytes {
		return
	}
	fmt.Fprintf(os.Stderr, format+" ", args...)
	if !deb.remainingKnown {
		return
	}
	if deb.remain < 0 {
		fmt.Fprintf(os.Stderr, "remaining byte count is negative! %d\n", deb.remain)
		return
	}
	data := make([]byte, deb.remain)
	n, _ := deb.r.peek(data)
	if n == 0 {
		os.Stderr.Write(empty)
		return
	}
	b := new(bytes.Buffer)
	fmt.Fprintf(b, "[%d]{\n", deb.remain)
	// Blanks until first byte
	lineLength := 0
	if n := len(data); n%10 != 0 {
		lineLength = 10 - n%10
		fmt.Fprintf(b, "\t%s", blanks[:lineLength*3])
	}
	// 10 bytes per line
	for len(data) > 0 {
		if lineLength == 0 {
			fmt.Fprint(b, "\t")
		}
		m := 10 - lineLength
		lineLength = 0
		if m > len(data) {
			m = len(data)
		}
		fmt.Fprintf(b, "% x\n", data[:m])
		data = data[m:]
	}
	fmt.Fprint(b, "}\n")
	os.Stderr.Write(b.Bytes())
}

// Debug prints a human-readable representation of the gob data read from r.
// It is a no-op unless debugging was enabled when the package was built.
func Debug(r io.Reader) {
	err := debug(r)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gob debug: %s\n", err)
	}
}

// debug implements Debug, but catches panics and returns
// them as errors to be printed by Debug.
func debug(r io.Reader) (err error) {
	defer catchError(&err)
	fmt.Fprintln(os.Stderr, "Start of debugging")
	deb := &debugger{
		r:        newPeekReader(r),
		wireType: make(map[typeId]*wireType),
		tmp:      make([]byte, 16),
	}
	if b, ok := r.(*bytes.Buffer); ok {
		deb.remain = b.Len()
		deb.remainingKnown = true
	}
	deb.gobStream()
	return
}

// note that we've consumed some bytes
func (deb *debugger) consumed(n int) {
	if deb.remainingKnown {
		deb.remain -= n
	}
}

// int64 decodes and returns the next integer, which must be present.
// Don't call this if you could be at EOF.
func (deb *debugger) int64() int64 {
	return toInt(deb.uint64())
}

// uint64 returns and decodes the next unsigned integer, which must be present.
// Don't call this if you could be at EOF.
// TODO: handle errors better.
func (deb *debugger) uint64() uint64 {
	n, w, err := decodeUintReader(deb.r, deb.tmp)
	if err != nil {
		errorf("debug: read error: %s", err)
	}
	deb.consumed(w)
	return n
}

// GobStream:
//	DelimitedMessage* (until EOF)
func (deb *debugger) gobStream() {
	// Make sure we're single-threaded through here.
	deb.mutex.Lock()
	defer deb.mutex.Unlock()

	for deb.delimitedMessage(0) {
	}
}

// DelimitedMessage:
//	uint(lengthOfMessage) Message
func (deb *debugger) delimitedMessage(indent tab) bool {
	for {
		n := deb.loadBlock(true)
		if n < 0 {
			return false
		}
		deb.dump("Delimited message of length %d", n)
		deb.message(indent)
	}
	return true
}

// loadBlock preps us to read a message
// of the length specified next in the input. It returns
// the length of the block. The argument tells whether
// an EOF is acceptable now.  If it is and one is found,
// the return value is negative.
func (deb *debugger) loadBlock(eofOK bool) int {
	n64, w, err := decodeUintReader(deb.r, deb.tmp) // deb.uint64 will error at EOF
	if err != nil {
		if eofOK && err == io.EOF {
			return -1
		}
		errorf("debug: unexpected error: %s", err)
	}
	deb.consumed(w)
	n := int(n64)
	if n < 0 {
		errorf("huge value for message length: %d", n64)
	}
	return int(n)
}

// Message:
//	TypeSequence TypedValue
// TypeSequence
//	(TypeDefinition DelimitedTypeDefinition*)?
// DelimitedTypeDefinition:
//	uint(lengthOfTypeDefinition) TypeDefinition
// TypedValue:
//	int(typeId) Value
func (deb *debugger) message(indent tab) bool {
	for {
		// Convert the uint64 to a signed integer typeId
		uid := deb.int64()
		id := typeId(uid)
		deb.dump("type id=%d", id)
		if id < 0 {
			deb.typeDefinition(indent, -id)
			n := deb.loadBlock(false)
			deb.dump("Message of length %d", n)
			continue
		} else {
			deb.value(indent, id)
			break
		}
	}
	return true
}

// Helper methods to make it easy to scan a type descriptor.

// common returns the CommonType at the input point.
func (deb *debugger) common() CommonType {
	fieldNum := -1
	name := ""
	id := typeId(0)
	for {
		delta := deb.delta(-1)
		if delta == 0 {
			break
		}
		fieldNum += delta
		switch fieldNum {
		case 0:
			name = deb.string()
		case 1:
			// Id typeId
			id = deb.typeId()
		default:
			errorf("corrupted CommonType")
		}
	}
	return CommonType{name, id}
}

// uint returns the unsigned int at the input point, as a uint (not uint64).
func (deb *debugger) uint() uint {
	return uint(deb.uint64())
}

// int returns the signed int at the input point, as an int (not int64).
func (deb *debugger) int() int {
	return int(deb.int64())
}

// typeId returns the type id at the input point.
func (deb *debugger) typeId() typeId {
	return typeId(deb.int64())
}

// string returns the string at the input point.
func (deb *debugger) string() string {
	x := int(deb.uint64())
	b := make([]byte, x)
	nb, _ := deb.r.Read(b)
	if nb != x {
		errorf("corrupted type")
	}
	deb.consumed(nb)
	return string(b)
}

// delta returns the field delta at the input point.  The expect argument,
// if non-negative, identifies what the value should be.
func (deb *debugger) delta(expect int) int {
	delta := int(deb.uint64())
	if delta < 0 || (expect >= 0 && delta != expect) {
		errorf("decode: corrupted type: delta %d expected %d", delta, expect)
	}
	return delta
}

// TypeDefinition:
//	[int(-typeId) (already read)] encodingOfWireType
func (deb *debugger) typeDefinition(indent tab, id typeId) {
	deb.dump("type definition for id %d", id)
	// Encoding is of a wireType. Decode the structure as usual
	fieldNum := -1
	wire := new(wireType)
	// A wireType defines a single field.
	delta := deb.delta(-1)
	fieldNum += delta
	switch fieldNum {
	case 0: // array type, one field of {{Common}, elem, length}
		// Field number 0 is CommonType
		deb.delta(1)
		com := deb.common()
		// Field number 1 is type Id of elem
		deb.delta(1)
		id := deb.typeId()
		// Field number 3 is length
		deb.delta(1)
		length := deb.int()
		wire.ArrayT = &arrayType{com, id, length}

	case 1: // slice type, one field of {{Common}, elem}
		// Field number 0 is CommonType
		deb.delta(1)
		com := deb.common()
		// Field number 1 is type Id of elem
		deb.delta(1)
		id := deb.typeId()
		wire.SliceT = &sliceType{com, id}

	case 2: // struct type, one field of {{Common}, []fieldType}
		// Field number 0 is CommonType
		deb.delta(1)
		com := deb.common()
		// Field number 1 is slice of FieldType
		deb.delta(1)
		numField := int(deb.uint())
		field := make([]*fieldType, numField)
		for i := 0; i < numField; i++ {
			field[i] = new(fieldType)
			deb.delta(1) // field 0 of fieldType: name
			field[i].Name = deb.string()
			deb.delta(1) // field 1 of fieldType: id
			field[i].Id = deb.typeId()
			deb.delta(0) // end of fieldType
		}
		wire.StructT = &structType{com, field}

	case 3: // map type, one field of {{Common}, key, elem}
		// Field number 0 is CommonType
		deb.delta(1)
		com := deb.common()
		// Field number 1 is type Id of key
		deb.delta(1)
		keyId := deb.typeId()
		// Field number 2 is type Id of elem
		deb.delta(1)
		elemId := deb.typeId()
		wire.MapT = &mapType{com, keyId, elemId}
	case 4: // GobEncoder type, one field of {{Common}}
		// Field number 0 is CommonType
		deb.delta(1)
		com := deb.common()
		wire.GobEncoderT = &gobEncoderType{com}
	case 5: // BinaryMarshaler type, one field of {{Common}}
		// Field number 0 is CommonType
		deb.delta(1)
		com := deb.common()
		wire.BinaryMarshalerT = &gobEncoderType{com}
	case 6: // TextMarshaler type, one field of {{Common}}
		// Field number 0 is CommonType
		deb.delta(1)
		com := deb.common()
		wire.TextMarshalerT = &gobEncoderType{com}
	default:
		errorf("bad field in type %d", fieldNum)
	}
	deb.printWireType(indent, wire)
	deb.delta(0) // end inner type (arrayType, etc.)
	deb.delta(0) // end wireType
	// Remember we've seen this type.
	deb.wireType[id] = wire
}

// Value:
//	SingletonValue | StructValue
func (deb *debugger) value(indent tab, id typeId) {
	wire, ok := deb.wireType[id]
	if ok && wire.StructT != nil {
		deb.structValue(indent, id)
	} else {
		deb.singletonValue(indent, id)
	}
}

// SingletonValue:
//	uint(0) FieldValue
func (deb *debugger) singletonValue(indent tab, id typeId) {
	deb.dump("Singleton value")
	// is it a builtin type?
	wire := deb.wireType[id]
	_, ok := builtinIdToType[id]
	if !ok && wire == nil {
		errorf("type id %d not defined", id)
	}
	m := deb.uint64()
	if m != 0 {
		errorf("expected zero; got %d", m)
	}
	deb.fieldValue(indent, id)
}

// InterfaceValue:
//	NilInterfaceValue | NonNilInterfaceValue
func (deb *debugger) interfaceValue(indent tab) {
	deb.dump("Start of interface value")
	if nameLen := deb.uint64(); nameLen == 0 {
		deb.nilInterfaceValue(indent)
	} else {
		deb.nonNilInterfaceValue(indent, int(nameLen))
	}
}

// NilInterfaceValue:
//	uint(0) [already read]
func (deb *debugger) nilInterfaceValue(indent tab) int {
	fmt.Fprintf(os.Stderr, "%snil interface\n", indent)
	return 0
}

// NonNilInterfaceValue:
//	ConcreteTypeName TypeSequence InterfaceContents
// ConcreteTypeName:
//	uint(lengthOfName) [already read=n] name
// InterfaceContents:
//	int(concreteTypeId) DelimitedValue
// DelimitedValue:
//	uint(length) Value
func (deb *debugger) nonNilInterfaceValue(indent tab, nameLen int) {
	// ConcreteTypeName
	b := make([]byte, nameLen)
	deb.r.Read(b) // TODO: CHECK THESE READS!!
	deb.consumed(nameLen)
	name := string(b)

	for {
		id := deb.typeId()
		if id < 0 {
			deb.typeDefinition(indent, -id)
			n := deb.loadBlock(false)
			deb.dump("Nested message of length %d", n)
		} else {
			// DelimitedValue
			x := deb.uint64() // in case we want to ignore the value; we don't.
			fmt.Fprintf(os.Stderr, "%sinterface value, type %q id=%d; valueLength %d\n", indent, name, id, x)
			deb.value(indent, id)
			break
		}
	}
}

// printCommonType prints a common type; used by printWireType.
func (deb *debugger) printCommonType(indent tab, kind string, common *CommonType) {
	indent.print()
	fmt.Fprintf(os.Stderr, "%s %q id=%d\n", kind, common.Name, common.Id)
}

// printWireType prints the contents of a wireType.
func (deb *debugger) printWireType(indent tab, wire *wireType) {
	fmt.Fprintf(os.Stderr, "%stype definition {\n", indent)
	indent++
	switch {
	case wire.ArrayT != nil:
		deb.printCommonType(indent, "array", &wire.ArrayT.CommonType)
		fmt.Fprintf(os.Stderr, "%slen %d\n", indent+1, wire.ArrayT.Len)
		fmt.Fprintf(os.Stderr, "%selemid %d\n", indent+1, wire.ArrayT.Elem)
	case wire.MapT != nil:
		deb.printCommonType(indent, "map", &wire.MapT.CommonType)
		fmt.Fprintf(os.Stderr, "%skey id=%d\n", indent+1, wire.MapT.Key)
		fmt.Fprintf(os.Stderr, "%selem id=%d\n", indent+1, wire.MapT.Elem)
	case wire.SliceT != nil:
		deb.printCommonType(indent, "slice", &wire.SliceT.CommonType)
		fmt.Fprintf(os.Stderr, "%selem id=%d\n", indent+1, wire.SliceT.Elem)
	case wire.StructT != nil:
		deb.printCommonType(indent, "struct", &wire.StructT.CommonType)
		for i, field := range wire.StructT.Field {
			fmt.Fprintf(os.Stderr, "%sfield %d:\t%s\tid=%d\n", indent+1, i, field.Name, field.Id)
		}
	case wire.GobEncoderT != nil:
		deb.printCommonType(indent, "GobEncoder", &wire.GobEncoderT.CommonType)
	}
	indent--
	fmt.Fprintf(os.Stderr, "%s}\n", indent)
}

// fieldValue prints a value of any type, such as a struct field.
// FieldValue:
//	builtinValue | ArrayValue | MapValue | SliceValue | StructValue | InterfaceValue
func (deb *debugger) fieldValue(indent tab, id typeId) {
	_, ok := builtinIdToType[id]
	if ok {
		if id == tInterface {
			deb.interfaceValue(indent)
		} else {
			deb.printBuiltin(indent, id)
		}
		return
	}
	wire, ok := deb.wireType[id]
	if !ok {
		errorf("type id %d not defined", id)
	}
	switch {
	case wire.ArrayT != nil:
		deb.arrayValue(indent, wire)
	case wire.MapT != nil:
		deb.mapValue(indent, wire)
	case wire.SliceT != nil:
		deb.sliceValue(indent, wire)
	case wire.StructT != nil:
		deb.structValue(indent, id)
	case wire.GobEncoderT != nil:
		deb.gobEncoderValue(indent, id)
	default:
		panic("bad wire type for field")
	}
}

// printBuiltin prints a value not of a fundamental type, that is,
// one whose type is known to gobs at bootstrap time.
func (deb *debugger) printBuiltin(indent tab, id typeId) {
	switch id {
	case tBool:
		x := deb.int64()
		if x == 0 {
			fmt.Fprintf(os.Stderr, "%sfalse\n", indent)
		} else {
			fmt.Fprintf(os.Stderr, "%strue\n", indent)
		}
	case tInt:
		x := deb.int64()
		fmt.Fprintf(os.Stderr, "%s%d\n", indent, x)
	case tUint:
		x := deb.int64()
		fmt.Fprintf(os.Stderr, "%s%d\n", indent, x)
	case tFloat:
		x := deb.uint64()
		fmt.Fprintf(os.Stderr, "%s%g\n", indent, floatFromBits(x))
	case tComplex:
		r := deb.uint64()
		i := deb.uint64()
		fmt.Fprintf(os.Stderr, "%s%g+%gi\n", indent, floatFromBits(r), floatFromBits(i))
	case tBytes:
		x := int(deb.uint64())
		b := make([]byte, x)
		deb.r.Read(b)
		deb.consumed(x)
		fmt.Fprintf(os.Stderr, "%s{% x}=%q\n", indent, b, b)
	case tString:
		x := int(deb.uint64())
		b := make([]byte, x)
		deb.r.Read(b)
		deb.consumed(x)
		fmt.Fprintf(os.Stderr, "%s%q\n", indent, b)
	default:
		panic("unknown builtin")
	}
}

// ArrayValue:
//	uint(n) FieldValue*n
func (deb *debugger) arrayValue(indent tab, wire *wireType) {
	elemId := wire.ArrayT.Elem
	u := deb.uint64()
	length := int(u)
	for i := 0; i < length; i++ {
		deb.fieldValue(indent, elemId)
	}
	if length != wire.ArrayT.Len {
		fmt.Fprintf(os.Stderr, "%s(wrong length for array: %d should be %d)\n", indent, length, wire.ArrayT.Len)
	}
}

// MapValue:
//	uint(n) (FieldValue FieldValue)*n  [n (key, value) pairs]
func (deb *debugger) mapValue(indent tab, wire *wireType) {
	keyId := wire.MapT.Key
	elemId := wire.MapT.Elem
	u := deb.uint64()
	length := int(u)
	for i := 0; i < length; i++ {
		deb.fieldValue(indent+1, keyId)
		deb.fieldValue(indent+1, elemId)
	}
}

// SliceValue:
//	uint(n) (n FieldValue)
func (deb *debugger) sliceValue(indent tab, wire *wireType) {
	elemId := wire.SliceT.Elem
	u := deb.uint64()
	length := int(u)
	deb.dump("Start of slice of length %d", length)

	for i := 0; i < length; i++ {
		deb.fieldValue(indent, elemId)
	}
}

// StructValue:
//	(uint(fieldDelta) FieldValue)*
func (deb *debugger) structValue(indent tab, id typeId) {
	deb.dump("Start of struct value of %q id=%d\n<<\n", id.name(), id)
	fmt.Fprintf(os.Stderr, "%s%s struct {\n", indent, id.name())
	wire, ok := deb.wireType[id]
	if !ok {
		errorf("type id %d not defined", id)
	}
	strct := wire.StructT
	fieldNum := -1
	indent++
	for {
		delta := deb.uint64()
		if delta == 0 { // struct terminator is zero delta fieldnum
			break
		}
		fieldNum += int(delta)
		if fieldNum < 0 || fieldNum >= len(strct.Field) {
			deb.dump("field number out of range: prevField=%d delta=%d", fieldNum-int(delta), delta)
			break
		}
		fmt.Fprintf(os.Stderr, "%sfield %d:\t%s\n", indent, fieldNum, wire.StructT.Field[fieldNum].Name)
		deb.fieldValue(indent+1, strct.Field[fieldNum].Id)
	}
	indent--
	fmt.Fprintf(os.Stderr, "%s} // end %s struct\n", indent, id.name())
	deb.dump(">> End of struct value of type %d %q", id, id.name())
}

// GobEncoderValue:
//	uint(n) byte*n
func (deb *debugger) gobEncoderValue(indent tab, id typeId) {
	len := deb.uint64()
	deb.dump("GobEncoder value of %q id=%d, length %d\n", id.name(), id, len)
	fmt.Fprintf(os.Stderr, "%s%s (implements GobEncoder)\n", indent, id.name())
	data := make([]byte, len)
	_, err := deb.r.Read(data)
	if err != nil {
		errorf("gobEncoder data read: %s", err)
	}
	fmt.Fprintf(os.Stderr, "%s[% .2x]\n", indent+1, data)
}
