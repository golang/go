// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

type Code interface {
	Marker() SyncMarker
	Value() int
}

type CodeVal int

func (c CodeVal) Marker() SyncMarker { return SyncVal }
func (c CodeVal) Value() int         { return int(c) }

const (
	ValBool CodeVal = iota
	ValString
	ValInt64
	ValBigInt
	ValBigRat
	ValBigFloat
)

type CodeType int

func (c CodeType) Marker() SyncMarker { return SyncType }
func (c CodeType) Value() int         { return int(c) }

const (
	TypeBasic CodeType = iota
	TypeNamed
	TypePointer
	TypeSlice
	TypeArray
	TypeChan
	TypeMap
	TypeSignature
	TypeStruct
	TypeInterface
	TypeUnion
	TypeTypeParam
)

type CodeObj int

func (c CodeObj) Marker() SyncMarker { return SyncCodeObj }
func (c CodeObj) Value() int         { return int(c) }

const (
	ObjAlias CodeObj = iota
	ObjConst
	ObjType
	ObjFunc
	ObjVar
	ObjStub
)
