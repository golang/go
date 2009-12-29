package gob

// This file is not normally included in the gob package.  Used only for debugging the package itself.

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
)

// Debug prints a human-readable representation of the gob data read from r.
func Debug(r io.Reader) { NewDecoder(r).debug() }

// debug is like Decode but just prints what it finds.  It should be safe even for corrupted data.
func (dec *Decoder) debug() {
	dec.state.err = nil
	for {
		// Read a count.
		var nbytes uint64
		nbytes, dec.state.err = decodeUintReader(dec.r, dec.countBuf[0:])
		if dec.state.err != nil {
			break
		}

		// Allocate the buffer.
		if nbytes > uint64(len(dec.buf)) {
			dec.buf = make([]byte, nbytes+1000)
		}
		dec.state.b = bytes.NewBuffer(dec.buf[0:nbytes])

		// Read the data
		_, dec.state.err = io.ReadFull(dec.r, dec.buf[0:nbytes])
		if dec.state.err != nil {
			if dec.state.err == os.EOF {
				dec.state.err = io.ErrUnexpectedEOF
			}
			break
		}

		// Receive a type id.
		id := typeId(decodeInt(dec.state))
		if dec.state.err != nil {
			break
		}

		// Is it a new type?
		if id < 0 { // 0 is the error state, handled above
			// If the id is negative, we have a type.
			fmt.Printf("new type id %d\n", -id)
			dec.printType(-id)
			if dec.state.err != nil {
				break
			}
			continue
		}

		fmt.Printf("type id %d\n", id)
		// No, it's a value.
		// Make sure the type has been defined already.
		_, ok := dec.wireType[id]
		if !ok {
			dec.state.err = errBadType
			break
		}
		fmt.Printf("\t%d bytes:\t% x\n", nbytes, dec.state.b.Bytes())
		dec.printData(0, id)
		break
	}
	if dec.state.err != nil {
		log.Stderr("debug:", dec.state.err)
	}
}

func (dec *Decoder) printType(id typeId) {
	// Have we already seen this type?  That's an error
	if _, alreadySeen := dec.wireType[id]; alreadySeen {
		dec.state.err = os.ErrorString("gob: duplicate type received")
		return
	}

	// Type:
	wire := new(wireType)
	dec.state.err = dec.decode(tWireType, wire)
	if dec.state.err == nil {
		printWireType(wire)
	}
	// Remember we've seen this type.
	dec.wireType[id] = wire
}

func printWireType(wire *wireType) {
	switch {
	case wire.array != nil:
		printCommonType("array", &wire.array.commonType)
		fmt.Printf("\tlen %d\n\telemid %d\n", wire.array.Len, wire.array.Elem)
	case wire.slice != nil:
		printCommonType("slice", &wire.slice.commonType)
		fmt.Printf("\telemid %d\n", wire.slice.Elem)
	case wire.strct != nil:
		printCommonType("struct", &wire.strct.commonType)
		for i, field := range wire.strct.field {
			fmt.Printf("\tfield %d:\t%s\tid=%d\n", i, field.name, field.id)
		}
	}
}

func printCommonType(kind string, common *commonType) {
	fmt.Printf("\t%s %s\n\tid: %d\n", kind, common.name, common._id)
}

func (dec *Decoder) printData(indent int, id typeId) {
	if dec.state.err != nil {
		return
	}
	// is it a builtin type?
	_, ok := builtinIdToType[id]
	if ok {
		dec.printBuiltin(indent, id)
		return
	}
	wire, ok := dec.wireType[id]
	if !ok {
		fmt.Printf("type id %d not defined\n", id)
		return
	}
	switch {
	case wire.array != nil:
		dec.printArray(indent+1, wire)
	case wire.slice != nil:
		dec.printSlice(indent+1, wire)
	case wire.strct != nil:
		dec.printStruct(indent+1, wire)
	}
}

func (dec *Decoder) printArray(indent int, wire *wireType) {
	elemId := wire.array.Elem
	n := int(decodeUint(dec.state))
	for i := 0; i < n && dec.state.err == nil; i++ {
		dec.printData(indent, elemId)
	}
	if n != wire.array.Len {
		tab(indent)
		fmt.Printf("(wrong length for array: %d should be %d)\n", n, wire.array.Len)
	}
}

func (dec *Decoder) printSlice(indent int, wire *wireType) {
	elemId := wire.slice.Elem
	n := int(decodeUint(dec.state))
	for i := 0; i < n && dec.state.err == nil; i++ {
		dec.printData(indent, elemId)
	}
}

func (dec *Decoder) printBuiltin(indent int, id typeId) {
	tab(indent)
	switch id {
	case tBool:
		if decodeInt(dec.state) == 0 {
			fmt.Printf("false")
		} else {
			fmt.Printf("true")
		}
	case tInt:
		fmt.Printf("%d", decodeInt(dec.state))
	case tUint:
		fmt.Printf("%d", decodeUint(dec.state))
	case tFloat:
		fmt.Printf("%g", floatFromBits(decodeUint(dec.state)))
	case tBytes:
		b := make([]byte, decodeUint(dec.state))
		dec.state.b.Read(b)
		fmt.Printf("% x", b)
	case tString:
		b := make([]byte, decodeUint(dec.state))
		dec.state.b.Read(b)
		fmt.Printf("%q", b)
	default:
		fmt.Print("unknown")
	}
	fmt.Print("\n")
}

func (dec *Decoder) printStruct(indent int, wire *wireType) {
	strct := wire.strct
	state := newDecodeState(dec.state.b)
	state.fieldnum = -1
	for state.err == nil {
		delta := int(decodeUint(state))
		if delta < 0 {
			dec.state.err = os.ErrorString("gob decode: corrupted data: negative delta")
			return
		}
		if state.err != nil || delta == 0 { // struct terminator is zero delta fieldnum
			return
		}
		fieldnum := state.fieldnum + delta
		if fieldnum < 0 || fieldnum >= len(strct.field) {
			dec.state.err = os.ErrorString("field number out of range")
			return
		}
		tab(indent)
		fmt.Printf("field %d:\n", fieldnum)
		dec.printData(indent+1, strct.field[fieldnum].id)
		state.fieldnum = fieldnum
	}
}

func tab(indent int) {
	for i := 0; i < indent; i++ {
		fmt.Print("\t")
	}
}
