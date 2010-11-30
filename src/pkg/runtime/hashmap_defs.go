// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go definitions of internal structures. Master is hashmap.[c,h]

package runtime

type hash_hash uintptr

type hash_entry struct {
	hash hash_hash
	key  byte // dwarf.c substitutes the real type
	val  byte // for key and val
}

type hash_subtable struct {
	power       uint8
	used        uint8
	datasize    uint8
	max_probes  uint8
	limit_bytes int16
	end         *hash_entry
	entry       hash_entry // TODO: [0]hash_entry
}

type hash struct {
	count       uint32
	datasize    uint8
	max_power   uint8
	max_probes  uint8
	indirectval uint8
	changes     int32
	data_hash   func(uint32, uintptr) hash_hash
	data_eq     func(uint32, uintptr, uintptr) uint32
	data_del    func(uint32, uintptr, uintptr)
	st          *hash_subtable
	keysize     uint32
	valsize     uint32
	datavo      uint32
	ko0         uint32
	vo0         uint32
	ko1         uint32
	vo1         uint32
	po1         uint32
	ko2         uint32
	vo2         uint32
	po2         uint32
	keyalg      *alg
	valalg      *alg
}
