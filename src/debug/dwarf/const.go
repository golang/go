// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Constants

package dwarf

import "strconv"

// An Attr identifies the attribute type in a DWARF Entry's Field.
type Attr uint32

const (
	AttrSibling        Attr = 0x01
	AttrLocation       Attr = 0x02
	AttrName           Attr = 0x03
	AttrOrdering       Attr = 0x09
	AttrByteSize       Attr = 0x0B
	AttrBitOffset      Attr = 0x0C
	AttrBitSize        Attr = 0x0D
	AttrStmtList       Attr = 0x10
	AttrLowpc          Attr = 0x11
	AttrHighpc         Attr = 0x12
	AttrLanguage       Attr = 0x13
	AttrDiscr          Attr = 0x15
	AttrDiscrValue     Attr = 0x16
	AttrVisibility     Attr = 0x17
	AttrImport         Attr = 0x18
	AttrStringLength   Attr = 0x19
	AttrCommonRef      Attr = 0x1A
	AttrCompDir        Attr = 0x1B
	AttrConstValue     Attr = 0x1C
	AttrContainingType Attr = 0x1D
	AttrDefaultValue   Attr = 0x1E
	AttrInline         Attr = 0x20
	AttrIsOptional     Attr = 0x21
	AttrLowerBound     Attr = 0x22
	AttrProducer       Attr = 0x25
	AttrPrototyped     Attr = 0x27
	AttrReturnAddr     Attr = 0x2A
	AttrStartScope     Attr = 0x2C
	AttrStrideSize     Attr = 0x2E
	AttrUpperBound     Attr = 0x2F
	AttrAbstractOrigin Attr = 0x31
	AttrAccessibility  Attr = 0x32
	AttrAddrClass      Attr = 0x33
	AttrArtificial     Attr = 0x34
	AttrBaseTypes      Attr = 0x35
	AttrCalling        Attr = 0x36
	AttrCount          Attr = 0x37
	AttrDataMemberLoc  Attr = 0x38
	AttrDeclColumn     Attr = 0x39
	AttrDeclFile       Attr = 0x3A
	AttrDeclLine       Attr = 0x3B
	AttrDeclaration    Attr = 0x3C
	AttrDiscrList      Attr = 0x3D
	AttrEncoding       Attr = 0x3E
	AttrExternal       Attr = 0x3F
	AttrFrameBase      Attr = 0x40
	AttrFriend         Attr = 0x41
	AttrIdentifierCase Attr = 0x42
	AttrMacroInfo      Attr = 0x43
	AttrNamelistItem   Attr = 0x44
	AttrPriority       Attr = 0x45
	AttrSegment        Attr = 0x46
	AttrSpecification  Attr = 0x47
	AttrStaticLink     Attr = 0x48
	AttrType           Attr = 0x49
	AttrUseLocation    Attr = 0x4A
	AttrVarParam       Attr = 0x4B
	AttrVirtuality     Attr = 0x4C
	AttrVtableElemLoc  Attr = 0x4D
	AttrAllocated      Attr = 0x4E
	AttrAssociated     Attr = 0x4F
	AttrDataLocation   Attr = 0x50
	AttrStride         Attr = 0x51
	AttrEntrypc        Attr = 0x52
	AttrUseUTF8        Attr = 0x53
	AttrExtension      Attr = 0x54
	AttrRanges         Attr = 0x55
	AttrTrampoline     Attr = 0x56
	AttrCallColumn     Attr = 0x57
	AttrCallFile       Attr = 0x58
	AttrCallLine       Attr = 0x59
	AttrDescription    Attr = 0x5A
)

var attrNames = [...]string{
	AttrSibling:        "Sibling",
	AttrLocation:       "Location",
	AttrName:           "Name",
	AttrOrdering:       "Ordering",
	AttrByteSize:       "ByteSize",
	AttrBitOffset:      "BitOffset",
	AttrBitSize:        "BitSize",
	AttrStmtList:       "StmtList",
	AttrLowpc:          "Lowpc",
	AttrHighpc:         "Highpc",
	AttrLanguage:       "Language",
	AttrDiscr:          "Discr",
	AttrDiscrValue:     "DiscrValue",
	AttrVisibility:     "Visibility",
	AttrImport:         "Import",
	AttrStringLength:   "StringLength",
	AttrCommonRef:      "CommonRef",
	AttrCompDir:        "CompDir",
	AttrConstValue:     "ConstValue",
	AttrContainingType: "ContainingType",
	AttrDefaultValue:   "DefaultValue",
	AttrInline:         "Inline",
	AttrIsOptional:     "IsOptional",
	AttrLowerBound:     "LowerBound",
	AttrProducer:       "Producer",
	AttrPrototyped:     "Prototyped",
	AttrReturnAddr:     "ReturnAddr",
	AttrStartScope:     "StartScope",
	AttrStrideSize:     "StrideSize",
	AttrUpperBound:     "UpperBound",
	AttrAbstractOrigin: "AbstractOrigin",
	AttrAccessibility:  "Accessibility",
	AttrAddrClass:      "AddrClass",
	AttrArtificial:     "Artificial",
	AttrBaseTypes:      "BaseTypes",
	AttrCalling:        "Calling",
	AttrCount:          "Count",
	AttrDataMemberLoc:  "DataMemberLoc",
	AttrDeclColumn:     "DeclColumn",
	AttrDeclFile:       "DeclFile",
	AttrDeclLine:       "DeclLine",
	AttrDeclaration:    "Declaration",
	AttrDiscrList:      "DiscrList",
	AttrEncoding:       "Encoding",
	AttrExternal:       "External",
	AttrFrameBase:      "FrameBase",
	AttrFriend:         "Friend",
	AttrIdentifierCase: "IdentifierCase",
	AttrMacroInfo:      "MacroInfo",
	AttrNamelistItem:   "NamelistItem",
	AttrPriority:       "Priority",
	AttrSegment:        "Segment",
	AttrSpecification:  "Specification",
	AttrStaticLink:     "StaticLink",
	AttrType:           "Type",
	AttrUseLocation:    "UseLocation",
	AttrVarParam:       "VarParam",
	AttrVirtuality:     "Virtuality",
	AttrVtableElemLoc:  "VtableElemLoc",
	AttrAllocated:      "Allocated",
	AttrAssociated:     "Associated",
	AttrDataLocation:   "DataLocation",
	AttrStride:         "Stride",
	AttrEntrypc:        "Entrypc",
	AttrUseUTF8:        "UseUTF8",
	AttrExtension:      "Extension",
	AttrRanges:         "Ranges",
	AttrTrampoline:     "Trampoline",
	AttrCallColumn:     "CallColumn",
	AttrCallFile:       "CallFile",
	AttrCallLine:       "CallLine",
	AttrDescription:    "Description",
}

func (a Attr) String() string {
	if int(a) < len(attrNames) {
		s := attrNames[a]
		if s != "" {
			return s
		}
	}
	return strconv.Itoa(int(a))
}

func (a Attr) GoString() string {
	if int(a) < len(attrNames) {
		s := attrNames[a]
		if s != "" {
			return "dwarf.Attr" + s
		}
	}
	return "dwarf.Attr(" + strconv.FormatInt(int64(a), 10) + ")"
}

// A format is a DWARF data encoding format.
type format uint32

const (
	// value formats
	formAddr        format = 0x01
	formDwarfBlock2 format = 0x03
	formDwarfBlock4 format = 0x04
	formData2       format = 0x05
	formData4       format = 0x06
	formData8       format = 0x07
	formString      format = 0x08
	formDwarfBlock  format = 0x09
	formDwarfBlock1 format = 0x0A
	formData1       format = 0x0B
	formFlag        format = 0x0C
	formSdata       format = 0x0D
	formStrp        format = 0x0E
	formUdata       format = 0x0F
	formRefAddr     format = 0x10
	formRef1        format = 0x11
	formRef2        format = 0x12
	formRef4        format = 0x13
	formRef8        format = 0x14
	formRefUdata    format = 0x15
	formIndirect    format = 0x16
	// The following are new in DWARF 4.
	formSecOffset   format = 0x17
	formExprloc     format = 0x18
	formFlagPresent format = 0x19
	formRefSig8     format = 0x20
	// Extensions for multi-file compression (.dwz)
	// http://www.dwarfstd.org/ShowIssue.php?issue=120604.1
	formGnuRefAlt  format = 0x1f20
	formGnuStrpAlt format = 0x1f21
)

// A Tag is the classification (the type) of an Entry.
type Tag uint32

const (
	TagArrayType              Tag = 0x01
	TagClassType              Tag = 0x02
	TagEntryPoint             Tag = 0x03
	TagEnumerationType        Tag = 0x04
	TagFormalParameter        Tag = 0x05
	TagImportedDeclaration    Tag = 0x08
	TagLabel                  Tag = 0x0A
	TagLexDwarfBlock          Tag = 0x0B
	TagMember                 Tag = 0x0D
	TagPointerType            Tag = 0x0F
	TagReferenceType          Tag = 0x10
	TagCompileUnit            Tag = 0x11
	TagStringType             Tag = 0x12
	TagStructType             Tag = 0x13
	TagSubroutineType         Tag = 0x15
	TagTypedef                Tag = 0x16
	TagUnionType              Tag = 0x17
	TagUnspecifiedParameters  Tag = 0x18
	TagVariant                Tag = 0x19
	TagCommonDwarfBlock       Tag = 0x1A
	TagCommonInclusion        Tag = 0x1B
	TagInheritance            Tag = 0x1C
	TagInlinedSubroutine      Tag = 0x1D
	TagModule                 Tag = 0x1E
	TagPtrToMemberType        Tag = 0x1F
	TagSetType                Tag = 0x20
	TagSubrangeType           Tag = 0x21
	TagWithStmt               Tag = 0x22
	TagAccessDeclaration      Tag = 0x23
	TagBaseType               Tag = 0x24
	TagCatchDwarfBlock        Tag = 0x25
	TagConstType              Tag = 0x26
	TagConstant               Tag = 0x27
	TagEnumerator             Tag = 0x28
	TagFileType               Tag = 0x29
	TagFriend                 Tag = 0x2A
	TagNamelist               Tag = 0x2B
	TagNamelistItem           Tag = 0x2C
	TagPackedType             Tag = 0x2D
	TagSubprogram             Tag = 0x2E
	TagTemplateTypeParameter  Tag = 0x2F
	TagTemplateValueParameter Tag = 0x30
	TagThrownType             Tag = 0x31
	TagTryDwarfBlock          Tag = 0x32
	TagVariantPart            Tag = 0x33
	TagVariable               Tag = 0x34
	TagVolatileType           Tag = 0x35
	// The following are new in DWARF 3.
	TagDwarfProcedure  Tag = 0x36
	TagRestrictType    Tag = 0x37
	TagInterfaceType   Tag = 0x38
	TagNamespace       Tag = 0x39
	TagImportedModule  Tag = 0x3A
	TagUnspecifiedType Tag = 0x3B
	TagPartialUnit     Tag = 0x3C
	TagImportedUnit    Tag = 0x3D
	TagMutableType     Tag = 0x3E // Later removed from DWARF.
	TagCondition       Tag = 0x3F
	TagSharedType      Tag = 0x40
	// The following are new in DWARF 4.
	TagTypeUnit            Tag = 0x41
	TagRvalueReferenceType Tag = 0x42
	TagTemplateAlias       Tag = 0x43
)

var tagNames = [...]string{
	TagArrayType:              "ArrayType",
	TagClassType:              "ClassType",
	TagEntryPoint:             "EntryPoint",
	TagEnumerationType:        "EnumerationType",
	TagFormalParameter:        "FormalParameter",
	TagImportedDeclaration:    "ImportedDeclaration",
	TagLabel:                  "Label",
	TagLexDwarfBlock:          "LexDwarfBlock",
	TagMember:                 "Member",
	TagPointerType:            "PointerType",
	TagReferenceType:          "ReferenceType",
	TagCompileUnit:            "CompileUnit",
	TagStringType:             "StringType",
	TagStructType:             "StructType",
	TagSubroutineType:         "SubroutineType",
	TagTypedef:                "Typedef",
	TagUnionType:              "UnionType",
	TagUnspecifiedParameters:  "UnspecifiedParameters",
	TagVariant:                "Variant",
	TagCommonDwarfBlock:       "CommonDwarfBlock",
	TagCommonInclusion:        "CommonInclusion",
	TagInheritance:            "Inheritance",
	TagInlinedSubroutine:      "InlinedSubroutine",
	TagModule:                 "Module",
	TagPtrToMemberType:        "PtrToMemberType",
	TagSetType:                "SetType",
	TagSubrangeType:           "SubrangeType",
	TagWithStmt:               "WithStmt",
	TagAccessDeclaration:      "AccessDeclaration",
	TagBaseType:               "BaseType",
	TagCatchDwarfBlock:        "CatchDwarfBlock",
	TagConstType:              "ConstType",
	TagConstant:               "Constant",
	TagEnumerator:             "Enumerator",
	TagFileType:               "FileType",
	TagFriend:                 "Friend",
	TagNamelist:               "Namelist",
	TagNamelistItem:           "NamelistItem",
	TagPackedType:             "PackedType",
	TagSubprogram:             "Subprogram",
	TagTemplateTypeParameter:  "TemplateTypeParameter",
	TagTemplateValueParameter: "TemplateValueParameter",
	TagThrownType:             "ThrownType",
	TagTryDwarfBlock:          "TryDwarfBlock",
	TagVariantPart:            "VariantPart",
	TagVariable:               "Variable",
	TagVolatileType:           "VolatileType",
	TagDwarfProcedure:         "DwarfProcedure",
	TagRestrictType:           "RestrictType",
	TagInterfaceType:          "InterfaceType",
	TagNamespace:              "Namespace",
	TagImportedModule:         "ImportedModule",
	TagUnspecifiedType:        "UnspecifiedType",
	TagPartialUnit:            "PartialUnit",
	TagImportedUnit:           "ImportedUnit",
	TagMutableType:            "MutableType",
	TagCondition:              "Condition",
	TagSharedType:             "SharedType",
	TagTypeUnit:               "TypeUnit",
	TagRvalueReferenceType:    "RvalueReferenceType",
	TagTemplateAlias:          "TemplateAlias",
}

func (t Tag) String() string {
	if int(t) < len(tagNames) {
		s := tagNames[t]
		if s != "" {
			return s
		}
	}
	return strconv.Itoa(int(t))
}

func (t Tag) GoString() string {
	if int(t) < len(tagNames) {
		s := tagNames[t]
		if s != "" {
			return "dwarf.Tag" + s
		}
	}
	return "dwarf.Tag(" + strconv.FormatInt(int64(t), 10) + ")"
}

// Location expression operators.
// The debug info encodes value locations like 8(R3)
// as a sequence of these op codes.
// This package does not implement full expressions;
// the opPlusUconst operator is expected by the type parser.
const (
	opAddr       = 0x03 /* 1 op, const addr */
	opDeref      = 0x06
	opConst1u    = 0x08 /* 1 op, 1 byte const */
	opConst1s    = 0x09 /*	" signed */
	opConst2u    = 0x0A /* 1 op, 2 byte const  */
	opConst2s    = 0x0B /*	" signed */
	opConst4u    = 0x0C /* 1 op, 4 byte const */
	opConst4s    = 0x0D /*	" signed */
	opConst8u    = 0x0E /* 1 op, 8 byte const */
	opConst8s    = 0x0F /*	" signed */
	opConstu     = 0x10 /* 1 op, LEB128 const */
	opConsts     = 0x11 /*	" signed */
	opDup        = 0x12
	opDrop       = 0x13
	opOver       = 0x14
	opPick       = 0x15 /* 1 op, 1 byte stack index */
	opSwap       = 0x16
	opRot        = 0x17
	opXderef     = 0x18
	opAbs        = 0x19
	opAnd        = 0x1A
	opDiv        = 0x1B
	opMinus      = 0x1C
	opMod        = 0x1D
	opMul        = 0x1E
	opNeg        = 0x1F
	opNot        = 0x20
	opOr         = 0x21
	opPlus       = 0x22
	opPlusUconst = 0x23 /* 1 op, ULEB128 addend */
	opShl        = 0x24
	opShr        = 0x25
	opShra       = 0x26
	opXor        = 0x27
	opSkip       = 0x2F /* 1 op, signed 2-byte constant */
	opBra        = 0x28 /* 1 op, signed 2-byte constant */
	opEq         = 0x29
	opGe         = 0x2A
	opGt         = 0x2B
	opLe         = 0x2C
	opLt         = 0x2D
	opNe         = 0x2E
	opLit0       = 0x30
	/* OpLitN = OpLit0 + N for N = 0..31 */
	opReg0 = 0x50
	/* OpRegN = OpReg0 + N for N = 0..31 */
	opBreg0 = 0x70 /* 1 op, signed LEB128 constant */
	/* OpBregN = OpBreg0 + N for N = 0..31 */
	opRegx       = 0x90 /* 1 op, ULEB128 register */
	opFbreg      = 0x91 /* 1 op, SLEB128 offset */
	opBregx      = 0x92 /* 2 op, ULEB128 reg; SLEB128 off */
	opPiece      = 0x93 /* 1 op, ULEB128 size of piece */
	opDerefSize  = 0x94 /* 1-byte size of data retrieved */
	opXderefSize = 0x95 /* 1-byte size of data retrieved */
	opNop        = 0x96
	/* next four new in Dwarf v3 */
	opPushObjAddr = 0x97
	opCall2       = 0x98 /* 2-byte offset of DIE */
	opCall4       = 0x99 /* 4-byte offset of DIE */
	opCallRef     = 0x9A /* 4- or 8- byte offset of DIE */
	/* 0xE0-0xFF reserved for user-specific */
)

// Basic type encodings -- the value for AttrEncoding in a TagBaseType Entry.
const (
	encAddress        = 0x01
	encBoolean        = 0x02
	encComplexFloat   = 0x03
	encFloat          = 0x04
	encSigned         = 0x05
	encSignedChar     = 0x06
	encUnsigned       = 0x07
	encUnsignedChar   = 0x08
	encImaginaryFloat = 0x09
)

// Statement program standard opcode encodings.
const (
	lnsCopy           = 1
	lnsAdvancePC      = 2
	lnsAdvanceLine    = 3
	lnsSetFile        = 4
	lnsSetColumn      = 5
	lnsNegateStmt     = 6
	lnsSetBasicBlock  = 7
	lnsConstAddPC     = 8
	lnsFixedAdvancePC = 9

	// DWARF 3
	lnsSetPrologueEnd   = 10
	lnsSetEpilogueBegin = 11
	lnsSetISA           = 12
)

// Statement program extended opcode encodings.
const (
	lneEndSequence = 1
	lneSetAddress  = 2
	lneDefineFile  = 3

	// DWARF 4
	lneSetDiscriminator = 4
)
