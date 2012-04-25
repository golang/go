// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

// Init is used by type Builder in exp/locale/collate/build/
// to create Collator instances.  It is for internal use only.
func Init(data interface{}) *Collator {
	init, ok := data.(tableInitializer)
	if !ok {
		return nil
	}
	t := &table{}
	t.index.index = init.TrieIndex()
	t.index.values = init.TrieValues()
	t.expandElem = init.ExpandElems()
	t.contractTries = init.ContractTries()
	t.contractElem = init.ContractElems()
	t.maxContractLen = init.MaxContractLen()
	return &Collator{t: t}
}

type tableInitializer interface {
	TrieIndex() []uint16
	TrieValues() []uint32
	ExpandElems() []uint32
	ContractTries() []struct{ l, h, n, i uint8 }
	ContractElems() []uint32
	MaxContractLen() int
}
