// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

// tableIndex holds information for constructing a table
// for a certain locale based on the main table.
type tableIndex struct {
	lookupOffset uint32
	valuesOffset uint32
}

func (t tableIndex) TrieIndex() []uint16 {
	return mainLookup[:]
}

func (t tableIndex) TrieValues() []uint32 {
	return mainValues[:]
}

func (t tableIndex) FirstBlockOffsets() (lookup, value uint16) {
	return uint16(t.lookupOffset), uint16(t.valuesOffset)
}

func (t tableIndex) ExpandElems() []uint32 {
	return mainExpandElem[:]
}

func (t tableIndex) ContractTries() []struct{ l, h, n, i uint8 } {
	return mainCTEntries[:]
}

func (t tableIndex) ContractElems() []uint32 {
	return mainContractElem[:]
}

func (t tableIndex) MaxContractLen() int {
	return 18 // TODO: generate
}

func (t tableIndex) VariableTop() uint32 {
	return varTop
}
