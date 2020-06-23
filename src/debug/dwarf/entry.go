// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DWARF debug information entry parser.
// An entry is a sequence of data items of a given format.
// The first word in the entry is an index into what DWARF
// calls the ``abbreviation table.''  An abbreviation is really
// just a type descriptor: it's an array of attribute tag/value format pairs.

package dwarf

import (
	"encoding/binary"
	"errors"
	"strconv"
)

// a single entry's description: a sequence of attributes
type abbrev struct {
	tag      Tag
	children bool
	field    []afield
}

type afield struct {
	attr  Attr
	fmt   format
	class Class
	val   int64 // for formImplicitConst
}

// a map from entry format ids to their descriptions
type abbrevTable map[uint32]abbrev

// ParseAbbrev returns the abbreviation table that starts at byte off
// in the .debug_abbrev section.
func (d *Data) parseAbbrev(off uint64, vers int) (abbrevTable, error) {
	if m, ok := d.abbrevCache[off]; ok {
		return m, nil
	}

	data := d.abbrev
	if off > uint64(len(data)) {
		data = nil
	} else {
		data = data[off:]
	}
	b := makeBuf(d, unknownFormat{}, "abbrev", 0, data)

	// Error handling is simplified by the buf getters
	// returning an endless stream of 0s after an error.
	m := make(abbrevTable)
	for {
		// Table ends with id == 0.
		id := uint32(b.uint())
		if id == 0 {
			break
		}

		// Walk over attributes, counting.
		n := 0
		b1 := b // Read from copy of b.
		b1.uint()
		b1.uint8()
		for {
			tag := b1.uint()
			fmt := b1.uint()
			if tag == 0 && fmt == 0 {
				break
			}
			if format(fmt) == formImplicitConst {
				b1.int()
			}
			n++
		}
		if b1.err != nil {
			return nil, b1.err
		}

		// Walk over attributes again, this time writing them down.
		var a abbrev
		a.tag = Tag(b.uint())
		a.children = b.uint8() != 0
		a.field = make([]afield, n)
		for i := range a.field {
			a.field[i].attr = Attr(b.uint())
			a.field[i].fmt = format(b.uint())
			a.field[i].class = formToClass(a.field[i].fmt, a.field[i].attr, vers, &b)
			if a.field[i].fmt == formImplicitConst {
				a.field[i].val = b.int()
			}
		}
		b.uint()
		b.uint()

		m[id] = a
	}
	if b.err != nil {
		return nil, b.err
	}
	d.abbrevCache[off] = m
	return m, nil
}

// attrIsExprloc indicates attributes that allow exprloc values that
// are encoded as block values in DWARF 2 and 3. See DWARF 4, Figure
// 20.
var attrIsExprloc = map[Attr]bool{
	AttrLocation:      true,
	AttrByteSize:      true,
	AttrBitOffset:     true,
	AttrBitSize:       true,
	AttrStringLength:  true,
	AttrLowerBound:    true,
	AttrReturnAddr:    true,
	AttrStrideSize:    true,
	AttrUpperBound:    true,
	AttrCount:         true,
	AttrDataMemberLoc: true,
	AttrFrameBase:     true,
	AttrSegment:       true,
	AttrStaticLink:    true,
	AttrUseLocation:   true,
	AttrVtableElemLoc: true,
	AttrAllocated:     true,
	AttrAssociated:    true,
	AttrDataLocation:  true,
	AttrStride:        true,
}

// attrPtrClass indicates the *ptr class of attributes that have
// encoding formSecOffset in DWARF 4 or formData* in DWARF 2 and 3.
var attrPtrClass = map[Attr]Class{
	AttrLocation:      ClassLocListPtr,
	AttrStmtList:      ClassLinePtr,
	AttrStringLength:  ClassLocListPtr,
	AttrReturnAddr:    ClassLocListPtr,
	AttrStartScope:    ClassRangeListPtr,
	AttrDataMemberLoc: ClassLocListPtr,
	AttrFrameBase:     ClassLocListPtr,
	AttrMacroInfo:     ClassMacPtr,
	AttrSegment:       ClassLocListPtr,
	AttrStaticLink:    ClassLocListPtr,
	AttrUseLocation:   ClassLocListPtr,
	AttrVtableElemLoc: ClassLocListPtr,
	AttrRanges:        ClassRangeListPtr,
	// The following are new in DWARF 5.
	AttrStrOffsetsBase: ClassStrOffsetsPtr,
	AttrAddrBase:       ClassAddrPtr,
	AttrRnglistsBase:   ClassRngListsPtr,
	AttrLoclistsBase:   ClassLocListPtr,
}

// formToClass returns the DWARF 4 Class for the given form. If the
// DWARF version is less then 4, it will disambiguate some forms
// depending on the attribute.
func formToClass(form format, attr Attr, vers int, b *buf) Class {
	switch form {
	default:
		b.error("cannot determine class of unknown attribute form")
		return 0

	case formIndirect:
		return ClassUnknown

	case formAddr, formAddrx, formAddrx1, formAddrx2, formAddrx3, formAddrx4:
		return ClassAddress

	case formDwarfBlock1, formDwarfBlock2, formDwarfBlock4, formDwarfBlock:
		// In DWARF 2 and 3, ClassExprLoc was encoded as a
		// block. DWARF 4 distinguishes ClassBlock and
		// ClassExprLoc, but there are no attributes that can
		// be both, so we also promote ClassBlock values in
		// DWARF 4 that should be ClassExprLoc in case
		// producers get this wrong.
		if attrIsExprloc[attr] {
			return ClassExprLoc
		}
		return ClassBlock

	case formData1, formData2, formData4, formData8, formSdata, formUdata, formData16, formImplicitConst:
		// In DWARF 2 and 3, ClassPtr was encoded as a
		// constant. Unlike ClassExprLoc/ClassBlock, some
		// DWARF 4 attributes need to distinguish Class*Ptr
		// from ClassConstant, so we only do this promotion
		// for versions 2 and 3.
		if class, ok := attrPtrClass[attr]; vers < 4 && ok {
			return class
		}
		return ClassConstant

	case formFlag, formFlagPresent:
		return ClassFlag

	case formRefAddr, formRef1, formRef2, formRef4, formRef8, formRefUdata, formRefSup4, formRefSup8:
		return ClassReference

	case formRefSig8:
		return ClassReferenceSig

	case formString, formStrp, formStrx, formStrpSup, formLineStrp, formStrx1, formStrx2, formStrx3, formStrx4:
		return ClassString

	case formSecOffset:
		// DWARF 4 defines four *ptr classes, but doesn't
		// distinguish them in the encoding. Disambiguate
		// these classes using the attribute.
		if class, ok := attrPtrClass[attr]; ok {
			return class
		}
		return ClassUnknown

	case formExprloc:
		return ClassExprLoc

	case formGnuRefAlt:
		return ClassReferenceAlt

	case formGnuStrpAlt:
		return ClassStringAlt

	case formLoclistx:
		return ClassLocList

	case formRnglistx:
		return ClassRngList
	}
}

// An entry is a sequence of attribute/value pairs.
type Entry struct {
	Offset   Offset // offset of Entry in DWARF info
	Tag      Tag    // tag (kind of Entry)
	Children bool   // whether Entry is followed by children
	Field    []Field
}

// A Field is a single attribute/value pair in an Entry.
//
// A value can be one of several "attribute classes" defined by DWARF.
// The Go types corresponding to each class are:
//
//    DWARF class       Go type        Class
//    -----------       -------        -----
//    address           uint64         ClassAddress
//    block             []byte         ClassBlock
//    constant          int64          ClassConstant
//    flag              bool           ClassFlag
//    reference
//      to info         dwarf.Offset   ClassReference
//      to type unit    uint64         ClassReferenceSig
//    string            string         ClassString
//    exprloc           []byte         ClassExprLoc
//    lineptr           int64          ClassLinePtr
//    loclistptr        int64          ClassLocListPtr
//    macptr            int64          ClassMacPtr
//    rangelistptr      int64          ClassRangeListPtr
//
// For unrecognized or vendor-defined attributes, Class may be
// ClassUnknown.
type Field struct {
	Attr  Attr
	Val   interface{}
	Class Class
}

// A Class is the DWARF 4 class of an attribute value.
//
// In general, a given attribute's value may take on one of several
// possible classes defined by DWARF, each of which leads to a
// slightly different interpretation of the attribute.
//
// DWARF version 4 distinguishes attribute value classes more finely
// than previous versions of DWARF. The reader will disambiguate
// coarser classes from earlier versions of DWARF into the appropriate
// DWARF 4 class. For example, DWARF 2 uses "constant" for constants
// as well as all types of section offsets, but the reader will
// canonicalize attributes in DWARF 2 files that refer to section
// offsets to one of the Class*Ptr classes, even though these classes
// were only defined in DWARF 3.
type Class int

const (
	// ClassUnknown represents values of unknown DWARF class.
	ClassUnknown Class = iota

	// ClassAddress represents values of type uint64 that are
	// addresses on the target machine.
	ClassAddress

	// ClassBlock represents values of type []byte whose
	// interpretation depends on the attribute.
	ClassBlock

	// ClassConstant represents values of type int64 that are
	// constants. The interpretation of this constant depends on
	// the attribute.
	ClassConstant

	// ClassExprLoc represents values of type []byte that contain
	// an encoded DWARF expression or location description.
	ClassExprLoc

	// ClassFlag represents values of type bool.
	ClassFlag

	// ClassLinePtr represents values that are an int64 offset
	// into the "line" section.
	ClassLinePtr

	// ClassLocListPtr represents values that are an int64 offset
	// into the "loclist" section.
	ClassLocListPtr

	// ClassMacPtr represents values that are an int64 offset into
	// the "mac" section.
	ClassMacPtr

	// ClassMacPtr represents values that are an int64 offset into
	// the "rangelist" section.
	ClassRangeListPtr

	// ClassReference represents values that are an Offset offset
	// of an Entry in the info section (for use with Reader.Seek).
	// The DWARF specification combines ClassReference and
	// ClassReferenceSig into class "reference".
	ClassReference

	// ClassReferenceSig represents values that are a uint64 type
	// signature referencing a type Entry.
	ClassReferenceSig

	// ClassString represents values that are strings. If the
	// compilation unit specifies the AttrUseUTF8 flag (strongly
	// recommended), the string value will be encoded in UTF-8.
	// Otherwise, the encoding is unspecified.
	ClassString

	// ClassReferenceAlt represents values of type int64 that are
	// an offset into the DWARF "info" section of an alternate
	// object file.
	ClassReferenceAlt

	// ClassStringAlt represents values of type int64 that are an
	// offset into the DWARF string section of an alternate object
	// file.
	ClassStringAlt

	// ClassAddrPtr represents values that are an int64 offset
	// into the "addr" section.
	ClassAddrPtr

	// ClassLocList represents values that are an int64 offset
	// into the "loclists" section.
	ClassLocList

	// ClassRngList represents values that are an int64 offset
	// from the base of the "rnglists" section.
	ClassRngList

	// ClassRngListsPtr represents values that are an int64 offset
	// into the "rnglists" section. These are used as the base for
	// ClassRngList values.
	ClassRngListsPtr

	// ClassStrOffsetsPtr represents values that are an int64
	// offset into the "str_offsets" section.
	ClassStrOffsetsPtr
)

//go:generate stringer -type=Class

func (i Class) GoString() string {
	return "dwarf." + i.String()
}

// Val returns the value associated with attribute Attr in Entry,
// or nil if there is no such attribute.
//
// A common idiom is to merge the check for nil return with
// the check that the value has the expected dynamic type, as in:
//	v, ok := e.Val(AttrSibling).(int64)
//
func (e *Entry) Val(a Attr) interface{} {
	if f := e.AttrField(a); f != nil {
		return f.Val
	}
	return nil
}

// AttrField returns the Field associated with attribute Attr in
// Entry, or nil if there is no such attribute.
func (e *Entry) AttrField(a Attr) *Field {
	for i, f := range e.Field {
		if f.Attr == a {
			return &e.Field[i]
		}
	}
	return nil
}

// An Offset represents the location of an Entry within the DWARF info.
// (See Reader.Seek.)
type Offset uint32

// Entry reads a single entry from buf, decoding
// according to the given abbreviation table.
func (b *buf) entry(cu *Entry, atab abbrevTable, ubase Offset, vers int) *Entry {
	off := b.off
	id := uint32(b.uint())
	if id == 0 {
		return &Entry{}
	}
	a, ok := atab[id]
	if !ok {
		b.error("unknown abbreviation table index")
		return nil
	}
	e := &Entry{
		Offset:   off,
		Tag:      a.tag,
		Children: a.children,
		Field:    make([]Field, len(a.field)),
	}
	for i := range e.Field {
		e.Field[i].Attr = a.field[i].attr
		e.Field[i].Class = a.field[i].class
		fmt := a.field[i].fmt
		if fmt == formIndirect {
			fmt = format(b.uint())
			e.Field[i].Class = formToClass(fmt, a.field[i].attr, vers, b)
		}
		var val interface{}
		switch fmt {
		default:
			b.error("unknown entry attr format 0x" + strconv.FormatInt(int64(fmt), 16))

		// address
		case formAddr:
			val = b.addr()
		case formAddrx, formAddrx1, formAddrx2, formAddrx3, formAddrx4:
			var off uint64
			switch fmt {
			case formAddrx:
				off = b.uint()
			case formAddrx1:
				off = uint64(b.uint8())
			case formAddrx2:
				off = uint64(b.uint16())
			case formAddrx3:
				off = uint64(b.uint24())
			case formAddrx4:
				off = uint64(b.uint32())
			}
			if len(b.dwarf.addr) == 0 {
				b.error("DW_FORM_addrx with no .debug_addr section")
			}
			if b.err != nil {
				return nil
			}
			addrsize := b.format.addrsize()
			if addrsize == 0 {
				b.error("unknown address size for DW_FORM_addrx")
			}
			off *= uint64(addrsize)

			// We have to adjust by the offset of the
			// compilation unit. This won't work if the
			// program uses Reader.Seek to skip over the
			// unit. Not much we can do about that.
			if cu != nil {
				cuOff, ok := cu.Val(AttrAddrBase).(int64)
				if ok {
					off += uint64(cuOff)
				}
			}

			if uint64(int(off)) != off {
				b.error("DW_FORM_addrx offset out of range")
			}

			b1 := makeBuf(b.dwarf, b.format, "addr", 0, b.dwarf.addr)
			b1.skip(int(off))
			val = b1.addr()
			if b1.err != nil {
				b.err = b1.err
				return nil
			}

		// block
		case formDwarfBlock1:
			val = b.bytes(int(b.uint8()))
		case formDwarfBlock2:
			val = b.bytes(int(b.uint16()))
		case formDwarfBlock4:
			val = b.bytes(int(b.uint32()))
		case formDwarfBlock:
			val = b.bytes(int(b.uint()))

		// constant
		case formData1:
			val = int64(b.uint8())
		case formData2:
			val = int64(b.uint16())
		case formData4:
			val = int64(b.uint32())
		case formData8:
			val = int64(b.uint64())
		case formData16:
			val = b.bytes(16)
		case formSdata:
			val = int64(b.int())
		case formUdata:
			val = int64(b.uint())
		case formImplicitConst:
			val = a.field[i].val

		// flag
		case formFlag:
			val = b.uint8() == 1
		// New in DWARF 4.
		case formFlagPresent:
			// The attribute is implicitly indicated as present, and no value is
			// encoded in the debugging information entry itself.
			val = true

		// reference to other entry
		case formRefAddr:
			vers := b.format.version()
			if vers == 0 {
				b.error("unknown version for DW_FORM_ref_addr")
			} else if vers == 2 {
				val = Offset(b.addr())
			} else {
				is64, known := b.format.dwarf64()
				if !known {
					b.error("unknown size for DW_FORM_ref_addr")
				} else if is64 {
					val = Offset(b.uint64())
				} else {
					val = Offset(b.uint32())
				}
			}
		case formRef1:
			val = Offset(b.uint8()) + ubase
		case formRef2:
			val = Offset(b.uint16()) + ubase
		case formRef4:
			val = Offset(b.uint32()) + ubase
		case formRef8:
			val = Offset(b.uint64()) + ubase
		case formRefUdata:
			val = Offset(b.uint()) + ubase

		// string
		case formString:
			val = b.string()
		case formStrp, formLineStrp:
			var off uint64 // offset into .debug_str
			is64, known := b.format.dwarf64()
			if !known {
				b.error("unknown size for DW_FORM_strp/line_strp")
			} else if is64 {
				off = b.uint64()
			} else {
				off = uint64(b.uint32())
			}
			if uint64(int(off)) != off {
				b.error("DW_FORM_strp/line_strp offset out of range")
			}
			if b.err != nil {
				return nil
			}
			var b1 buf
			if fmt == formStrp {
				b1 = makeBuf(b.dwarf, b.format, "str", 0, b.dwarf.str)
			} else {
				if len(b.dwarf.lineStr) == 0 {
					b.error("DW_FORM_line_strp with no .debug_line_str section")
				}
				b1 = makeBuf(b.dwarf, b.format, "line_str", 0, b.dwarf.lineStr)
			}
			b1.skip(int(off))
			val = b1.string()
			if b1.err != nil {
				b.err = b1.err
				return nil
			}
		case formStrx, formStrx1, formStrx2, formStrx3, formStrx4:
			var off uint64
			switch fmt {
			case formStrx:
				off = b.uint()
			case formStrx1:
				off = uint64(b.uint8())
			case formStrx2:
				off = uint64(b.uint16())
			case formStrx3:
				off = uint64(b.uint24())
			case formStrx4:
				off = uint64(b.uint32())
			}
			if len(b.dwarf.strOffsets) == 0 {
				b.error("DW_FORM_strx with no .debug_str_offsets section")
			}
			is64, known := b.format.dwarf64()
			if !known {
				b.error("unknown offset size for DW_FORM_strx")
			}
			if b.err != nil {
				return nil
			}
			if is64 {
				off *= 8
			} else {
				off *= 4
			}

			// We have to adjust by the offset of the
			// compilation unit. This won't work if the
			// program uses Reader.Seek to skip over the
			// unit. Not much we can do about that.
			if cu != nil {
				cuOff, ok := cu.Val(AttrStrOffsetsBase).(int64)
				if ok {
					off += uint64(cuOff)
				}
			}

			if uint64(int(off)) != off {
				b.error("DW_FORM_strx offset out of range")
			}

			b1 := makeBuf(b.dwarf, b.format, "str_offsets", 0, b.dwarf.strOffsets)
			b1.skip(int(off))
			if is64 {
				off = b1.uint64()
			} else {
				off = uint64(b1.uint32())
			}
			if b1.err != nil {
				b.err = b1.err
				return nil
			}
			if uint64(int(off)) != off {
				b.error("DW_FORM_strx indirect offset out of range")
			}
			b1 = makeBuf(b.dwarf, b.format, "str", 0, b.dwarf.str)
			b1.skip(int(off))
			val = b1.string()
			if b1.err != nil {
				b.err = b1.err
				return nil
			}
		case formStrpSup:
			is64, known := b.format.dwarf64()
			if !known {
				b.error("unknown size for DW_FORM_strp_sup")
			} else if is64 {
				val = b.uint64()
			} else {
				val = b.uint32()
			}

		// lineptr, loclistptr, macptr, rangelistptr
		// New in DWARF 4, but clang can generate them with -gdwarf-2.
		// Section reference, replacing use of formData4 and formData8.
		case formSecOffset, formGnuRefAlt, formGnuStrpAlt:
			is64, known := b.format.dwarf64()
			if !known {
				b.error("unknown size for form 0x" + strconv.FormatInt(int64(fmt), 16))
			} else if is64 {
				val = int64(b.uint64())
			} else {
				val = int64(b.uint32())
			}

		// exprloc
		// New in DWARF 4.
		case formExprloc:
			val = b.bytes(int(b.uint()))

		// reference
		// New in DWARF 4.
		case formRefSig8:
			// 64-bit type signature.
			val = b.uint64()
		case formRefSup4:
			val = b.uint32()
		case formRefSup8:
			val = b.uint64()

		// loclist
		case formLoclistx:
			val = b.uint()

		// rnglist
		case formRnglistx:
			val = b.uint()
		}
		e.Field[i].Val = val
	}
	if b.err != nil {
		return nil
	}
	return e
}

// A Reader allows reading Entry structures from a DWARF ``info'' section.
// The Entry structures are arranged in a tree. The Reader's Next function
// return successive entries from a pre-order traversal of the tree.
// If an entry has children, its Children field will be true, and the children
// follow, terminated by an Entry with Tag 0.
type Reader struct {
	b            buf
	d            *Data
	err          error
	unit         int
	lastChildren bool   // .Children of last entry returned by Next
	lastSibling  Offset // .Val(AttrSibling) of last entry returned by Next
	cu           *Entry // current compilation unit
}

// Reader returns a new Reader for Data.
// The reader is positioned at byte offset 0 in the DWARF ``info'' section.
func (d *Data) Reader() *Reader {
	r := &Reader{d: d}
	r.Seek(0)
	return r
}

// AddressSize returns the size in bytes of addresses in the current compilation
// unit.
func (r *Reader) AddressSize() int {
	return r.d.unit[r.unit].asize
}

// ByteOrder returns the byte order in the current compilation unit.
func (r *Reader) ByteOrder() binary.ByteOrder {
	return r.b.order
}

// Seek positions the Reader at offset off in the encoded entry stream.
// Offset 0 can be used to denote the first entry.
func (r *Reader) Seek(off Offset) {
	d := r.d
	r.err = nil
	r.lastChildren = false
	if off == 0 {
		if len(d.unit) == 0 {
			return
		}
		u := &d.unit[0]
		r.unit = 0
		r.b = makeBuf(r.d, u, "info", u.off, u.data)
		r.cu = nil
		return
	}

	i := d.offsetToUnit(off)
	if i == -1 {
		r.err = errors.New("offset out of range")
		return
	}
	if i != r.unit {
		r.cu = nil
	}
	u := &d.unit[i]
	r.unit = i
	r.b = makeBuf(r.d, u, "info", off, u.data[off-u.off:])
}

// maybeNextUnit advances to the next unit if this one is finished.
func (r *Reader) maybeNextUnit() {
	for len(r.b.data) == 0 && r.unit+1 < len(r.d.unit) {
		r.unit++
		u := &r.d.unit[r.unit]
		r.b = makeBuf(r.d, u, "info", u.off, u.data)
		r.cu = nil
	}
}

// Next reads the next entry from the encoded entry stream.
// It returns nil, nil when it reaches the end of the section.
// It returns an error if the current offset is invalid or the data at the
// offset cannot be decoded as a valid Entry.
func (r *Reader) Next() (*Entry, error) {
	if r.err != nil {
		return nil, r.err
	}
	r.maybeNextUnit()
	if len(r.b.data) == 0 {
		return nil, nil
	}
	u := &r.d.unit[r.unit]
	e := r.b.entry(r.cu, u.atable, u.base, u.vers)
	if r.b.err != nil {
		r.err = r.b.err
		return nil, r.err
	}
	if e != nil {
		r.lastChildren = e.Children
		if r.lastChildren {
			r.lastSibling, _ = e.Val(AttrSibling).(Offset)
		}
		if e.Tag == TagCompileUnit || e.Tag == TagPartialUnit {
			r.cu = e
		}
	} else {
		r.lastChildren = false
	}
	return e, nil
}

// SkipChildren skips over the child entries associated with
// the last Entry returned by Next. If that Entry did not have
// children or Next has not been called, SkipChildren is a no-op.
func (r *Reader) SkipChildren() {
	if r.err != nil || !r.lastChildren {
		return
	}

	// If the last entry had a sibling attribute,
	// that attribute gives the offset of the next
	// sibling, so we can avoid decoding the
	// child subtrees.
	if r.lastSibling >= r.b.off {
		r.Seek(r.lastSibling)
		return
	}

	for {
		e, err := r.Next()
		if err != nil || e == nil || e.Tag == 0 {
			break
		}
		if e.Children {
			r.SkipChildren()
		}
	}
}

// clone returns a copy of the reader. This is used by the typeReader
// interface.
func (r *Reader) clone() typeReader {
	return r.d.Reader()
}

// offset returns the current buffer offset. This is used by the
// typeReader interface.
func (r *Reader) offset() Offset {
	return r.b.off
}

// SeekPC returns the Entry for the compilation unit that includes pc,
// and positions the reader to read the children of that unit.  If pc
// is not covered by any unit, SeekPC returns ErrUnknownPC and the
// position of the reader is undefined.
//
// Because compilation units can describe multiple regions of the
// executable, in the worst case SeekPC must search through all the
// ranges in all the compilation units. Each call to SeekPC starts the
// search at the compilation unit of the last call, so in general
// looking up a series of PCs will be faster if they are sorted. If
// the caller wishes to do repeated fast PC lookups, it should build
// an appropriate index using the Ranges method.
func (r *Reader) SeekPC(pc uint64) (*Entry, error) {
	unit := r.unit
	for i := 0; i < len(r.d.unit); i++ {
		if unit >= len(r.d.unit) {
			unit = 0
		}
		r.err = nil
		r.lastChildren = false
		r.unit = unit
		u := &r.d.unit[unit]
		r.b = makeBuf(r.d, u, "info", u.off, u.data)
		e, err := r.Next()
		if err != nil {
			return nil, err
		}
		ranges, err := r.d.Ranges(e)
		if err != nil {
			return nil, err
		}
		for _, pcs := range ranges {
			if pcs[0] <= pc && pc < pcs[1] {
				return e, nil
			}
		}
		unit++
	}
	return nil, ErrUnknownPC
}

// Ranges returns the PC ranges covered by e, a slice of [low,high) pairs.
// Only some entry types, such as TagCompileUnit or TagSubprogram, have PC
// ranges; for others, this will return nil with no error.
func (d *Data) Ranges(e *Entry) ([][2]uint64, error) {
	var ret [][2]uint64

	low, lowOK := e.Val(AttrLowpc).(uint64)

	var high uint64
	var highOK bool
	highField := e.AttrField(AttrHighpc)
	if highField != nil {
		switch highField.Class {
		case ClassAddress:
			high, highOK = highField.Val.(uint64)
		case ClassConstant:
			off, ok := highField.Val.(int64)
			if ok {
				high = low + uint64(off)
				highOK = true
			}
		}
	}

	if lowOK && highOK {
		ret = append(ret, [2]uint64{low, high})
	}

	ranges, rangesOK := e.Val(AttrRanges).(int64)
	if rangesOK && d.ranges != nil {
		// The initial base address is the lowpc attribute
		// of the enclosing compilation unit.
		// Although DWARF specifies the lowpc attribute,
		// comments in gdb/dwarf2read.c say that some versions
		// of GCC use the entrypc attribute, so we check that too.
		var cu *Entry
		if e.Tag == TagCompileUnit {
			cu = e
		} else {
			i := d.offsetToUnit(e.Offset)
			if i == -1 {
				return nil, errors.New("no unit for entry")
			}
			u := &d.unit[i]
			b := makeBuf(d, u, "info", u.off, u.data)
			cu = b.entry(nil, u.atable, u.base, u.vers)
			if b.err != nil {
				return nil, b.err
			}
		}

		var base uint64
		if cuEntry, cuEntryOK := cu.Val(AttrEntrypc).(uint64); cuEntryOK {
			base = cuEntry
		} else if cuLow, cuLowOK := cu.Val(AttrLowpc).(uint64); cuLowOK {
			base = cuLow
		}

		u := &d.unit[d.offsetToUnit(e.Offset)]
		buf := makeBuf(d, u, "ranges", Offset(ranges), d.ranges[ranges:])
		for len(buf.data) > 0 {
			low = buf.addr()
			high = buf.addr()

			if low == 0 && high == 0 {
				break
			}

			if low == ^uint64(0)>>uint((8-u.addrsize())*8) {
				base = high
			} else {
				ret = append(ret, [2]uint64{base + low, base + high})
			}
		}
	}

	return ret, nil
}
