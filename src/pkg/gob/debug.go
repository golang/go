package gob

// This file is not normally included in the gob package.  Used only for debugging the package itself.
// Add debug.go to the files listed in the Makefile to add Debug to the gob package.
// Except for reading uints, it is an implementation of a reader that is independent of
// the one implemented by Decoder.

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
func (p *peekReader) Read(b []byte) (n int, err os.Error) {
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
func (p *peekReader) peek(b []byte) (n int, err os.Error) {
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
			e = os.EOF
		}
	}
	return n, e
}

// dump prints the next nBytes of the input.
// It arranges to print the output aligned from call to
// call, to make it easy to see what has been consumed.
func (deb *debugger) dump(nBytes int, format string, args ...interface{}) {
	if !dumpBytes {
		return
	}
	fmt.Fprintf(os.Stderr, format+" ", args...)
	if nBytes < 0 {
		fmt.Fprintf(os.Stderr, "nbytes is negative! %d\n", nBytes)
		return
	}
	data := make([]byte, nBytes)
	n, _ := deb.r.peek(data)
	if n == 0 {
		os.Stderr.Write(empty)
		return
	}
	b := new(bytes.Buffer)
	fmt.Fprint(b, "{\n")
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

type debugger struct {
	mutex    sync.Mutex
	r        *peekReader
	wireType map[typeId]*wireType
	tmp      []byte // scratch space for decoding uints.
}

// Debug prints a human-readable representation of the gob data read from r.
func Debug(r io.Reader) {
	fmt.Fprintln(os.Stderr, "Start of debugging")
	deb := &debugger{
		r:        newPeekReader(r),
		wireType: make(map[typeId]*wireType),
		tmp:      make([]byte, 16),
	}
	deb.gobStream()
}

// toInt turns an encoded uint64 into an int, according to the marshaling rules.
func toInt(x uint64) int64 {
	i := int64(x >> 1)
	if x&1 != 0 {
		i = ^i
	}
	return i
}

// readInt returns the next int, which must be present,
// and the number of bytes it consumed.
// Don't call this if you could be at EOF.
func (deb *debugger) readInt() (i int64, w int) {
	var u uint64
	u, w = deb.readUint()
	return toInt(u), w
}

// readUint returns the next uint, which must be present.
// and the number of bytes it consumed.
// Don't call this if you could be at EOF.
// TODO: handle errors better.
func (deb *debugger) readUint() (x uint64, w int) {
	n, w, err := decodeUintReader(deb.r, deb.tmp)
	if err != nil {
		errorf("debug: read error: %s", err)
	}
	return n, w
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
		deb.dump(int(n), "Message of length %d", n)
		deb.message(indent, n)
	}
	return true
}

// loadBlock preps us to read a message
// of the length specified next in the input. It returns
// the length of the block. The argument tells whether
// an EOF is acceptable now.  If it is and one is found,
// the return value is negative.
func (deb *debugger) loadBlock(eofOK bool) int {
	n64, _, err := decodeUintReader(deb.r, deb.tmp)
	if err != nil {
		if eofOK && err == os.EOF {
			return -1
		}
		errorf("debug: unexpected error: %s", err)
	}
	n := int(n64)
	if n < 0 {
		errorf("huge value for message length: %d", n64)
	}
	return n
}

// Message:
//	TypeSequence TypedValue
// TypeSequence
//	(TypeDefinition DelimitedTypeDefinition*)?
// DelimitedTypeDefinition:
//	uint(lengthOfTypeDefinition) TypeDefinition
// TypedValue:
//	int(typeId) Value
func (deb *debugger) message(indent tab, n int) bool {
	for {
		// Convert the uint64 to a signed integer typeId
		uid, w := deb.readInt()
		id := typeId(uid)
		n -= w
		deb.dump(n, "type id=%d", id)
		if id < 0 {
			n -= deb.typeDefinition(indent, -id, n)
			n = deb.loadBlock(false)
			deb.dump(n, "Message of length %d", n)
			continue
		} else {
			deb.value(indent, id, n)
			break
		}
	}
	return true
}

// TypeDefinition:
//	[int(-typeId) (already read)] encodingOfWireType
func (deb *debugger) typeDefinition(indent tab, id typeId, n int) int {
	deb.dump(n, "type definition for id %d", id)
	// Encoding is of a wireType. Decode the structure as usual
	fieldNum := -1
	m := 0

	// Closures to make it easy to scan.

	// Read a uint from the input
	getUint := func() uint {
		i, w := deb.readUint()
		m += w
		n -= w
		return uint(i)
	}
	// Read an int from the input
	getInt := func() int {
		i, w := deb.readInt()
		m += w
		n -= w
		return int(i)
	}
	// Read a string from the input
	getString := func() string {
		u, w := deb.readUint()
		x := int(u)
		m += w
		n -= w
		b := make([]byte, x)
		nb, _ := deb.r.Read(b)
		if nb != x {
			errorf("corrupted type")
		}
		m += x
		n -= x
		return string(b)
	}
	// Read a typeId from the input
	getTypeId := func() typeId {
		return typeId(getInt())
	}
	// Read a delta from the input.
	getDelta := func(expect int) int {
		u, w := deb.readUint()
		m += w
		n -= w
		delta := int(u)
		if delta < 0 || (expect >= 0 && delta != expect) {
			errorf("gob decode: corrupted type: delta %d expected %d", delta, expect)
		}
		return int(u)
	}
	// Read a CommonType from the input
	common := func() CommonType {
		fieldNum := -1
		name := ""
		id := typeId(0)
		for {
			delta := getDelta(-1)
			if delta == 0 {
				break
			}
			fieldNum += delta
			switch fieldNum {
			case 0:
				name = getString()
			case 1:
				// Id typeId
				id = getTypeId()
			default:
				errorf("corrupted CommonType")
			}
		}
		return CommonType{name, id}
	}

	wire := new(wireType)
	// A wireType defines a single field.
	delta := getDelta(-1)
	fieldNum += delta
	switch fieldNum {
	case 0: // array type, one field of {{Common}, elem, length}
		// Field number 0 is CommonType
		getDelta(1)
		com := common()
		// Field number 1 is type Id of elem
		getDelta(1)
		id := getTypeId()
		// Field number 3 is length
		getDelta(1)
		length := getInt()
		wire.ArrayT = &arrayType{com, id, length}

	case 1: // slice type, one field of {{Common}, elem}
		// Field number 0 is CommonType
		getDelta(1)
		com := common()
		// Field number 1 is type Id of elem
		getDelta(1)
		id := getTypeId()
		wire.SliceT = &sliceType{com, id}

	case 2: // struct type, one field of {{Common}, []fieldType}
		// Field number 0 is CommonType
		getDelta(1)
		com := common()
		// Field number 1 is slice of FieldType
		getDelta(1)
		numField := int(getUint())
		field := make([]*fieldType, numField)
		for i := 0; i < numField; i++ {
			field[i] = new(fieldType)
			getDelta(1) // field 0 of fieldType: name
			field[i].Name = getString()
			getDelta(1) // field 1 of fieldType: id
			field[i].Id = getTypeId()
			getDelta(0) // end of fieldType
		}
		wire.StructT = &structType{com, field}

	case 3: // map type, one field of {{Common}, key, elem}
		// Field number 0 is CommonType
		getDelta(1)
		com := common()
		// Field number 1 is type Id of key
		getDelta(1)
		keyId := getTypeId()
		wire.SliceT = &sliceType{com, id}
		// Field number 2 is type Id of elem
		getDelta(1)
		elemId := getTypeId()
		wire.MapT = &mapType{com, keyId, elemId}
	default:
		errorf("bad field in type %d", fieldNum)
	}
	deb.printWireType(indent, wire)
	getDelta(0) // end inner type (arrayType, etc.)
	getDelta(0) // end wireType
	// Remember we've seen this type.
	deb.wireType[id] = wire
	return m
}


// Value:
//	SingletonValue | StructValue
func (deb *debugger) value(indent tab, id typeId, n int) int {
	wire, ok := deb.wireType[id]
	if ok && wire.StructT != nil {
		return deb.structValue(indent, id, n)
	}
	return deb.singletonValue(indent, id, n)
}

// SingletonValue:
//	uint(0) FieldValue
func (deb *debugger) singletonValue(indent tab, id typeId, n int) int {
	deb.dump(n, "Singleton value")
	// is it a builtin type?
	wire := deb.wireType[id]
	_, ok := builtinIdToType[id]
	if !ok && wire == nil {
		errorf("type id %d not defined", id)
	}
	m, w := deb.readUint()
	if m != 0 {
		errorf("expected zero; got %d", n)
	}
	return w + deb.fieldValue(indent, id, n-w)
}

// InterfaceValue:
//	NilInterfaceValue | NonNilInterfaceValue
func (deb *debugger) interfaceValue(indent tab, n int) int {
	deb.dump(n, "Start of interface value")
	nameLen, w := deb.readUint()
	n -= w
	if n == 0 {
		return w + deb.nilInterfaceValue(indent)
	}
	return w + deb.nonNilInterfaceValue(indent, int(nameLen), n)
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
func (deb *debugger) nonNilInterfaceValue(indent tab, nameLen, n int) int {
	// ConcreteTypeName
	b := make([]byte, nameLen)
	deb.r.Read(b) // TODO: CHECK THESE READS!!
	w := nameLen
	n -= nameLen
	name := string(b)
	fmt.Fprintf(os.Stderr, "%sinterface value, type %q length %d\n", indent, name, n)

	for {
		x, width := deb.readInt()
		n -= w
		w += width
		id := typeId(x)
		if id < 0 {
			deb.typeDefinition(indent, -id, n)
			n = deb.loadBlock(false)
			deb.dump(n, "Message of length %d", n)
		} else {
			// DelimitedValue
			x, width := deb.readUint() // in case we want to ignore the value; we don't.
			n -= w
			w += width
			fmt.Fprintf(os.Stderr, "%sinterface value, type %q id=%d; length %d\n", indent, name, id, x)
			ZZ := w + deb.value(indent, id, int(x))
			return ZZ
		}
	}
	panic("not reached")
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
	}
	indent--
	fmt.Fprintf(os.Stderr, "%s}\n", indent)
}

// fieldValue prints a value of any type, such as a struct field.
// FieldValue:
//	builtinValue | ArrayValue | MapValue | SliceValue | StructValue | InterfaceValue
func (deb *debugger) fieldValue(indent tab, id typeId, n int) int {
	_, ok := builtinIdToType[id]
	if ok {
		if id == tInterface {
			return deb.interfaceValue(indent, n)
		}
		return deb.printBuiltin(indent, id, n)
	}
	wire, ok := deb.wireType[id]
	if !ok {
		errorf("type id %d not defined", id)
	}
	switch {
	case wire.ArrayT != nil:
		return deb.arrayValue(indent, wire, n)
	case wire.MapT != nil:
		return deb.mapValue(indent, wire, n)
	case wire.SliceT != nil:
		return deb.sliceValue(indent, wire, n)
	case wire.StructT != nil:
		return deb.structValue(indent, id, n)
	}
	panic("unreached")
}

// printBuiltin prints a value not of a fundamental type, that is,
// one whose type is known to gobs at bootstrap time.
func (deb *debugger) printBuiltin(indent tab, id typeId, n int) int {
	switch id {
	case tBool:
		x, w := deb.readInt()
		if x == 0 {
			fmt.Fprintf(os.Stderr, "%sfalse\n", indent)
		} else {
			fmt.Fprintf(os.Stderr, "%strue\n", indent)
		}
		return w
	case tInt:
		x, w := deb.readInt()
		fmt.Fprintf(os.Stderr, "%s%d\n", indent, x)
		return w
	case tUint:
		x, w := deb.readInt()
		fmt.Fprintf(os.Stderr, "%s%d\n", indent, x)
		return w
	case tFloat:
		x, w := deb.readUint()
		fmt.Fprintf(os.Stderr, "%s%g\n", indent, floatFromBits(x))
		return w
	case tBytes:
		x, w := deb.readUint()
		b := make([]byte, x)
		deb.r.Read(b)
		fmt.Fprintf(os.Stderr, "%s{% x}=%q\n", indent, b, b)
		return w + int(x)
	case tString:
		x, w := deb.readUint()
		b := make([]byte, x)
		deb.r.Read(b)
		fmt.Fprintf(os.Stderr, "%s%q\n", indent, b)
		return w + int(x)
	default:
		fmt.Print("unknown\n")
	}
	panic("unknown builtin")
}


// ArrayValue:
//	uint(n) FieldValue*n
func (deb *debugger) arrayValue(indent tab, wire *wireType, n int) int {
	elemId := wire.ArrayT.Elem
	u, w := deb.readUint()
	length := int(u)
	for i := 0; i < length; i++ {
		w += deb.fieldValue(indent, elemId, n-w)
	}
	if length != wire.ArrayT.Len {
		fmt.Fprintf(os.Stderr, "%s(wrong length for array: %d should be %d)\n", indent, length, wire.ArrayT.Len)
	}
	return w
}

// MapValue:
//	uint(n) (FieldValue FieldValue)*n  [n (key, value) pairs]
func (deb *debugger) mapValue(indent tab, wire *wireType, n int) int {
	keyId := wire.MapT.Key
	elemId := wire.MapT.Elem
	u, w := deb.readUint()
	length := int(u)
	for i := 0; i < length; i++ {
		w += deb.fieldValue(indent+1, keyId, n-w)
		w += deb.fieldValue(indent+1, elemId, n-w)
	}
	return w
}

// SliceValue:
//	uint(n) (n FieldValue)
func (deb *debugger) sliceValue(indent tab, wire *wireType, n int) int {
	elemId := wire.SliceT.Elem
	u, w := deb.readUint()
	length := int(u)
	for i := 0; i < length; i++ {
		w += deb.fieldValue(indent, elemId, n-w)
	}
	return w
}

// StructValue:
//	(uint(fieldDelta) FieldValue)*
func (deb *debugger) structValue(indent tab, id typeId, n int) int {
	deb.dump(n, "Start of struct value of %q id=%d\n<<\n", id.name(), id)
	fmt.Fprintf(os.Stderr, "%s%s struct {\n", indent, id.name())
	wire, ok := deb.wireType[id]
	if !ok {
		errorf("type id %d not defined", id)
	}
	strct := wire.StructT
	fieldNum := -1
	indent++
	w := 0
	for {
		delta, wid := deb.readUint()
		w += wid
		n -= wid
		if delta == 0 { // struct terminator is zero delta fieldnum
			break
		}
		fieldNum += int(delta)
		if fieldNum < 0 || fieldNum >= len(strct.Field) {
			deb.dump(n, "field number out of range: prevField=%d delta=%d", fieldNum-int(delta), delta)
			break
		}
		fmt.Fprintf(os.Stderr, "%sfield %d:\t%s\n", indent, fieldNum, wire.StructT.Field[fieldNum].Name)
		wid = deb.fieldValue(indent+1, strct.Field[fieldNum].Id, n)
		w += wid
		n -= wid
	}
	indent--
	fmt.Fprintf(os.Stderr, "%s} // end %s struct\n", indent, id.name())
	deb.dump(n, ">> End of struct value of type %d %q", id, id.name())
	return w
}
