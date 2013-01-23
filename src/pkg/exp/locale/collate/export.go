// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

// Init is for internal use only.
func Init(data interface{}) Weigher {
	init, ok := data.(tableInitializer)
	if !ok {
		return nil
	}
	t := &table{}
	loff, voff := init.FirstBlockOffsets()
	t.index.index = init.TrieIndex()
	t.index.index0 = t.index.index[blockSize*int(loff):]
	t.index.values = init.TrieValues()
	t.index.values0 = t.index.values[blockSize*int(voff):]
	t.expandElem = init.ExpandElems()
	t.contractTries = init.ContractTries()
	t.contractElem = init.ContractElems()
	t.maxContractLen = init.MaxContractLen()
	t.variableTop = init.VariableTop()
	return t
}

type tableInitializer interface {
	TrieIndex() []uint16
	TrieValues() []uint32
	FirstBlockOffsets() (lookup, value uint16)
	ExpandElems() []uint32
	ContractTries() []struct{ l, h, n, i uint8 }
	ContractElems() []uint32
	MaxContractLen() int
	VariableTop() uint32
}
