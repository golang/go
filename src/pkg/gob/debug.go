package gob

// This file is not normally included in the gob package.  Used only for debugging the package itself.
// Add debug.go to the files listed in the Makefile to add Debug to the gob package.

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"reflect"
	"runtime"
)

var dump = false // If true, print the remaining bytes in the input buffer at each item.

// Init installs the debugging facility. If this file is not compiled in the
// package, the test in codec_test.go is a no-op.
func init() {
	debugFunc = Debug
}

// Debug prints a human-readable representation of the gob data read from r.
func Debug(r io.Reader) {
	defer func() {
		if e := recover(); e != nil {
			if _, ok := e.(runtime.Error); ok {
				panic(e)
			}
			fmt.Printf("error during debugging: %v\n", e)
		}
	}()
	NewDecoder(r).debug()
}

// debugRecv is like recv but prints what it sees.
func (dec *Decoder) debugRecv() {
	if dec.byteBuffer != nil && dec.byteBuffer.Len() != 0 {
		fmt.Printf("error in recv: %d bytes left in input buffer\n", dec.byteBuffer.Len())
		return
	}
	// Read a count.
	var nbytes uint64
	nbytes, dec.err = decodeUintReader(dec.r, dec.countBuf[0:])
	if dec.err != nil {
		fmt.Printf("receiver error on count: %s\n", dec.err)
		return
	}
	// Allocate the buffer.
	if nbytes > uint64(len(dec.buf)) {
		dec.buf = make([]byte, nbytes+1000)
	}
	dec.byteBuffer = bytes.NewBuffer(dec.buf[0:nbytes])

	// Read the data
	_, dec.err = io.ReadFull(dec.r, dec.buf[0:nbytes])
	if dec.err != nil {
		fmt.Printf("receiver error on data: %s\n", dec.err)
		if dec.err == os.EOF {
			dec.err = io.ErrUnexpectedEOF
		}
		return
	}
	if dump {
		fmt.Printf("received %d bytes:\n\t% x\n", nbytes, dec.byteBuffer.Bytes())
	}
}


// debug is like Decode but just prints what it finds.  It should be safe even for corrupted data.
func (dec *Decoder) debug() {
	// Make sure we're single-threaded through here.
	dec.mutex.Lock()
	defer dec.mutex.Unlock()

	dec.err = nil
	dec.debugRecv()
	if dec.err != nil {
		return
	}
	dec.debugFromBuffer(0, false)
}

// printFromBuffer prints the next value.  The buffer contains data, but it may
// be a type descriptor and we may need to load more data to see the value;
// printType takes care of that.
func (dec *Decoder) debugFromBuffer(indent int, countPresent bool) {
	for dec.state.b.Len() > 0 {
		// Receive a type id.
		id := typeId(dec.state.decodeInt())

		// Is it a new type?
		if id < 0 { // 0 is the error state, handled above
			// If the id is negative, we have a type.
			dec.debugRecvType(-id)
			if dec.err != nil {
				break
			}
			continue
		}

		// No, it's a value.
		// Make sure the type has been defined already or is a builtin type (for
		// top-level singleton values).
		if dec.wireType[id] == nil && builtinIdToType[id] == nil {
			dec.err = errBadType
			break
		}
		if countPresent {
			dec.state.decodeUint()
		}
		dec.debugPrint(indent, id)
		break
	}
}

func (dec *Decoder) debugRecvType(id typeId) {
	// Have we already seen this type?  That's an error
	if _, alreadySeen := dec.wireType[id]; alreadySeen {
		dec.err = os.ErrorString("gob: duplicate type received")
		return
	}

	// Type:
	wire := new(wireType)
	dec.err = dec.decode(tWireType, reflect.NewValue(wire))
	if dec.err == nil {
		printWireType(wire)
	}
	// Remember we've seen this type.
	dec.wireType[id] = wire

	// Load the next parcel.
	dec.debugRecv()
}

func printWireType(wire *wireType) {
	fmt.Printf("type definition {\n")
	switch {
	case wire.ArrayT != nil:
		printCommonType("array", &wire.ArrayT.CommonType)
		fmt.Printf("\tlen %d\n\telemid %d\n", wire.ArrayT.Len, wire.ArrayT.Elem)
	case wire.MapT != nil:
		printCommonType("map", &wire.MapT.CommonType)
		fmt.Printf("\tkeyid %d\n", wire.MapT.Key)
		fmt.Printf("\telemid %d\n", wire.MapT.Elem)
	case wire.SliceT != nil:
		printCommonType("slice", &wire.SliceT.CommonType)
		fmt.Printf("\telemid %d\n", wire.SliceT.Elem)
	case wire.StructT != nil:
		printCommonType("struct", &wire.StructT.CommonType)
		for i, field := range wire.StructT.Field {
			fmt.Printf("\tfield %d:\t%s\tid=%d\n", i, field.Name, field.Id)
		}
	}
	fmt.Printf("}\n")
}

func printCommonType(kind string, common *CommonType) {
	fmt.Printf("\t%s %q\n\tid: %d\n", kind, common.Name, common.Id)
}

func (dec *Decoder) debugPrint(indent int, id typeId) {
	wire, ok := dec.wireType[id]
	if ok && wire.StructT != nil {
		dec.debugStruct(indent+1, id, wire)
	} else {
		dec.debugSingle(indent+1, id, wire)
	}
}

func (dec *Decoder) debugSingle(indent int, id typeId, wire *wireType) {
	// is it a builtin type?
	_, ok := builtinIdToType[id]
	if !ok && wire == nil {
		errorf("type id %d not defined\n", id)
	}
	dec.state.decodeUint()
	dec.printItem(indent, id)
}

func (dec *Decoder) printItem(indent int, id typeId) {
	if dump {
		fmt.Printf("print item %d bytes: % x\n", dec.state.b.Len(), dec.state.b.Bytes())
	}
	_, ok := builtinIdToType[id]
	if ok {
		dec.printBuiltin(indent, id)
		return
	}
	wire, ok := dec.wireType[id]
	if !ok {
		errorf("type id %d not defined\n", id)
	}
	switch {
	case wire.ArrayT != nil:
		dec.printArray(indent, wire)
	case wire.MapT != nil:
		dec.printMap(indent, wire)
	case wire.SliceT != nil:
		dec.printSlice(indent, wire)
	case wire.StructT != nil:
		dec.debugStruct(indent, id, wire)
	}
}

func (dec *Decoder) printArray(indent int, wire *wireType) {
	elemId := wire.ArrayT.Elem
	n := int(dec.state.decodeUint())
	for i := 0; i < n && dec.err == nil; i++ {
		dec.printItem(indent, elemId)
	}
	if n != wire.ArrayT.Len {
		tab(indent)
		fmt.Printf("(wrong length for array: %d should be %d)\n", n, wire.ArrayT.Len)
	}
}

func (dec *Decoder) printMap(indent int, wire *wireType) {
	keyId := wire.MapT.Key
	elemId := wire.MapT.Elem
	n := int(dec.state.decodeUint())
	for i := 0; i < n && dec.err == nil; i++ {
		dec.printItem(indent, keyId)
		dec.printItem(indent+1, elemId)
	}
}

func (dec *Decoder) printSlice(indent int, wire *wireType) {
	elemId := wire.SliceT.Elem
	n := int(dec.state.decodeUint())
	for i := 0; i < n && dec.err == nil; i++ {
		dec.printItem(indent, elemId)
	}
}

func (dec *Decoder) printBuiltin(indent int, id typeId) {
	tab(indent)
	switch id {
	case tBool:
		if dec.state.decodeInt() == 0 {
			fmt.Printf("false\n")
		} else {
			fmt.Printf("true\n")
		}
	case tInt:
		fmt.Printf("%d\n", dec.state.decodeInt())
	case tUint:
		fmt.Printf("%d\n", dec.state.decodeUint())
	case tFloat:
		fmt.Printf("%g\n", floatFromBits(dec.state.decodeUint()))
	case tBytes:
		b := make([]byte, dec.state.decodeUint())
		dec.state.b.Read(b)
		fmt.Printf("% x\n", b)
	case tString:
		b := make([]byte, dec.state.decodeUint())
		dec.state.b.Read(b)
		fmt.Printf("%q\n", b)
	case tInterface:
		b := make([]byte, dec.state.decodeUint())
		dec.state.b.Read(b)
		if len(b) == 0 {
			fmt.Printf("nil interface")
		} else {
			fmt.Printf("interface value; type %q\n", b)
			dec.debugFromBuffer(indent, true)
		}
	default:
		fmt.Print("unknown\n")
	}
}

func (dec *Decoder) debugStruct(indent int, id typeId, wire *wireType) {
	tab(indent)
	fmt.Printf("%s struct {\n", id.name())
	strct := wire.StructT
	state := newDecodeState(dec, dec.state.b)
	state.fieldnum = -1
	for dec.err == nil {
		delta := int(state.decodeUint())
		if delta < 0 {
			errorf("gob decode: corrupted data: negative delta")
		}
		if delta == 0 { // struct terminator is zero delta fieldnum
			break
		}
		fieldNum := state.fieldnum + delta
		if fieldNum < 0 || fieldNum >= len(strct.Field) {
			errorf("field number out of range")
			break
		}
		tab(indent)
		fmt.Printf("%s(%d):\n", wire.StructT.Field[fieldNum].Name, fieldNum)
		dec.printItem(indent+1, strct.Field[fieldNum].Id)
		state.fieldnum = fieldNum
	}
	tab(indent)
	fmt.Printf(" } // end %s struct\n", id.name())
}

func tab(indent int) {
	for i, w := 0, 0; i < indent; i += w {
		w = 10
		if i+w > indent {
			w = indent - i
		}
		fmt.Print("\t\t\t\t\t\t\t\t\t\t"[:w])
	}
}
