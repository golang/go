// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpack

import (
	"fmt"
)

// headerFieldTable implements a list of HeaderFields.
// This is used to implement the static and dynamic tables.
type headerFieldTable struct {
	// For static tables, entries are never evicted.
	//
	// For dynamic tables, entries are evicted from ents[0] and added to the end.
	// Each entry has a unique id that starts at one and increments for each
	// entry that is added. This unique id is stable across evictions, meaning
	// it can be used as a pointer to a specific entry. As in hpack, unique ids
	// are 1-based. The unique id for ents[k] is k + evictCount + 1.
	//
	// Zero is not a valid unique id.
	//
	// evictCount should not overflow in any remotely practical situation. In
	// practice, we will have one dynamic table per HTTP/2 connection. If we
	// assume a very powerful server that handles 1M QPS per connection and each
	// request adds (then evicts) 100 entries from the table, it would still take
	// 2M years for evictCount to overflow.
	ents       []HeaderField
	evictCount uint64

	// byName maps a HeaderField name to the unique id of the newest entry with
	// the same name. See above for a definition of "unique id".
	byName map[string]uint64

	// byNameValue maps a HeaderField name/value pair to the unique id of the newest
	// entry with the same name and value. See above for a definition of "unique id".
	byNameValue map[pairNameValue]uint64
}

type pairNameValue struct {
	name, value string
}

func (t *headerFieldTable) init() {
	t.byName = make(map[string]uint64)
	t.byNameValue = make(map[pairNameValue]uint64)
}

// len reports the number of entries in the table.
func (t *headerFieldTable) len() int {
	return len(t.ents)
}

// addEntry adds a new entry.
func (t *headerFieldTable) addEntry(f HeaderField) {
	id := uint64(t.len()) + t.evictCount + 1
	t.byName[f.Name] = id
	t.byNameValue[pairNameValue{f.Name, f.Value}] = id
	t.ents = append(t.ents, f)
}

// evictOldest evicts the n oldest entries in the table.
func (t *headerFieldTable) evictOldest(n int) {
	if n > t.len() {
		panic(fmt.Sprintf("evictOldest(%v) on table with %v entries", n, t.len()))
	}
	for k := 0; k < n; k++ {
		f := t.ents[k]
		id := t.evictCount + uint64(k) + 1
		if t.byName[f.Name] == id {
			delete(t.byName, f.Name)
		}
		if p := (pairNameValue{f.Name, f.Value}); t.byNameValue[p] == id {
			delete(t.byNameValue, p)
		}
	}
	copy(t.ents, t.ents[n:])
	for k := t.len() - n; k < t.len(); k++ {
		t.ents[k] = HeaderField{} // so strings can be garbage collected
	}
	t.ents = t.ents[:t.len()-n]
	if t.evictCount+uint64(n) < t.evictCount {
		panic("evictCount overflow")
	}
	t.evictCount += uint64(n)
}

// search finds f in the table. If there is no match, i is 0.
// If both name and value match, i is the matched index and nameValueMatch
// becomes true. If only name matches, i points to that index and
// nameValueMatch becomes false.
//
// The returned index is a 1-based HPACK index. For dynamic tables, HPACK says
// that index 1 should be the newest entry, but t.ents[0] is the oldest entry,
// meaning t.ents is reversed for dynamic tables. Hence, when t is a dynamic
// table, the return value i actually refers to the entry t.ents[t.len()-i].
//
// All tables are assumed to be a dynamic tables except for the global
// staticTable pointer.
//
// See Section 2.3.3.
func (t *headerFieldTable) search(f HeaderField) (i uint64, nameValueMatch bool) {
	if !f.Sensitive {
		if id := t.byNameValue[pairNameValue{f.Name, f.Value}]; id != 0 {
			return t.idToIndex(id), true
		}
	}
	if id := t.byName[f.Name]; id != 0 {
		return t.idToIndex(id), false
	}
	return 0, false
}

// idToIndex converts a unique id to an HPACK index.
// See Section 2.3.3.
func (t *headerFieldTable) idToIndex(id uint64) uint64 {
	if id <= t.evictCount {
		panic(fmt.Sprintf("id (%v) <= evictCount (%v)", id, t.evictCount))
	}
	k := id - t.evictCount - 1 // convert id to an index t.ents[k]
	if t != staticTable {
		return uint64(t.len()) - k // dynamic table
	}
	return k + 1
}

func pair(name, value string) HeaderField {
	return HeaderField{Name: name, Value: value}
}

// http://tools.ietf.org/html/draft-ietf-httpbis-header-compression-07#appendix-B
var staticTable = newStaticTable()

func newStaticTable() *headerFieldTable {
	t := &headerFieldTable{}
	t.init()
	t.addEntry(pair(":authority", ""))
	t.addEntry(pair(":method", "GET"))
	t.addEntry(pair(":method", "POST"))
	t.addEntry(pair(":path", "/"))
	t.addEntry(pair(":path", "/index.html"))
	t.addEntry(pair(":scheme", "http"))
	t.addEntry(pair(":scheme", "https"))
	t.addEntry(pair(":status", "200"))
	t.addEntry(pair(":status", "204"))
	t.addEntry(pair(":status", "206"))
	t.addEntry(pair(":status", "304"))
	t.addEntry(pair(":status", "400"))
	t.addEntry(pair(":status", "404"))
	t.addEntry(pair(":status", "500"))
	t.addEntry(pair("accept-charset", ""))
	t.addEntry(pair("accept-encoding", "gzip, deflate"))
	t.addEntry(pair("accept-language", ""))
	t.addEntry(pair("accept-ranges", ""))
	t.addEntry(pair("accept", ""))
	t.addEntry(pair("access-control-allow-origin", ""))
	t.addEntry(pair("age", ""))
	t.addEntry(pair("allow", ""))
	t.addEntry(pair("authorization", ""))
	t.addEntry(pair("cache-control", ""))
	t.addEntry(pair("content-disposition", ""))
	t.addEntry(pair("content-encoding", ""))
	t.addEntry(pair("content-language", ""))
	t.addEntry(pair("content-length", ""))
	t.addEntry(pair("content-location", ""))
	t.addEntry(pair("content-range", ""))
	t.addEntry(pair("content-type", ""))
	t.addEntry(pair("cookie", ""))
	t.addEntry(pair("date", ""))
	t.addEntry(pair("etag", ""))
	t.addEntry(pair("expect", ""))
	t.addEntry(pair("expires", ""))
	t.addEntry(pair("from", ""))
	t.addEntry(pair("host", ""))
	t.addEntry(pair("if-match", ""))
	t.addEntry(pair("if-modified-since", ""))
	t.addEntry(pair("if-none-match", ""))
	t.addEntry(pair("if-range", ""))
	t.addEntry(pair("if-unmodified-since", ""))
	t.addEntry(pair("last-modified", ""))
	t.addEntry(pair("link", ""))
	t.addEntry(pair("location", ""))
	t.addEntry(pair("max-forwards", ""))
	t.addEntry(pair("proxy-authenticate", ""))
	t.addEntry(pair("proxy-authorization", ""))
	t.addEntry(pair("range", ""))
	t.addEntry(pair("referer", ""))
	t.addEntry(pair("refresh", ""))
	t.addEntry(pair("retry-after", ""))
	t.addEntry(pair("server", ""))
	t.addEntry(pair("set-cookie", ""))
	t.addEntry(pair("strict-transport-security", ""))
	t.addEntry(pair("transfer-encoding", ""))
	t.addEntry(pair("user-agent", ""))
	t.addEntry(pair("vary", ""))
	t.addEntry(pair("via", ""))
	t.addEntry(pair("www-authenticate", ""))
	return t
}

var huffmanCodes = [256]uint32{
	0x1ff8,
	0x7fffd8,
	0xfffffe2,
	0xfffffe3,
	0xfffffe4,
	0xfffffe5,
	0xfffffe6,
	0xfffffe7,
	0xfffffe8,
	0xffffea,
	0x3ffffffc,
	0xfffffe9,
	0xfffffea,
	0x3ffffffd,
	0xfffffeb,
	0xfffffec,
	0xfffffed,
	0xfffffee,
	0xfffffef,
	0xffffff0,
	0xffffff1,
	0xffffff2,
	0x3ffffffe,
	0xffffff3,
	0xffffff4,
	0xffffff5,
	0xffffff6,
	0xffffff7,
	0xffffff8,
	0xffffff9,
	0xffffffa,
	0xffffffb,
	0x14,
	0x3f8,
	0x3f9,
	0xffa,
	0x1ff9,
	0x15,
	0xf8,
	0x7fa,
	0x3fa,
	0x3fb,
	0xf9,
	0x7fb,
	0xfa,
	0x16,
	0x17,
	0x18,
	0x0,
	0x1,
	0x2,
	0x19,
	0x1a,
	0x1b,
	0x1c,
	0x1d,
	0x1e,
	0x1f,
	0x5c,
	0xfb,
	0x7ffc,
	0x20,
	0xffb,
	0x3fc,
	0x1ffa,
	0x21,
	0x5d,
	0x5e,
	0x5f,
	0x60,
	0x61,
	0x62,
	0x63,
	0x64,
	0x65,
	0x66,
	0x67,
	0x68,
	0x69,
	0x6a,
	0x6b,
	0x6c,
	0x6d,
	0x6e,
	0x6f,
	0x70,
	0x71,
	0x72,
	0xfc,
	0x73,
	0xfd,
	0x1ffb,
	0x7fff0,
	0x1ffc,
	0x3ffc,
	0x22,
	0x7ffd,
	0x3,
	0x23,
	0x4,
	0x24,
	0x5,
	0x25,
	0x26,
	0x27,
	0x6,
	0x74,
	0x75,
	0x28,
	0x29,
	0x2a,
	0x7,
	0x2b,
	0x76,
	0x2c,
	0x8,
	0x9,
	0x2d,
	0x77,
	0x78,
	0x79,
	0x7a,
	0x7b,
	0x7ffe,
	0x7fc,
	0x3ffd,
	0x1ffd,
	0xffffffc,
	0xfffe6,
	0x3fffd2,
	0xfffe7,
	0xfffe8,
	0x3fffd3,
	0x3fffd4,
	0x3fffd5,
	0x7fffd9,
	0x3fffd6,
	0x7fffda,
	0x7fffdb,
	0x7fffdc,
	0x7fffdd,
	0x7fffde,
	0xffffeb,
	0x7fffdf,
	0xffffec,
	0xffffed,
	0x3fffd7,
	0x7fffe0,
	0xffffee,
	0x7fffe1,
	0x7fffe2,
	0x7fffe3,
	0x7fffe4,
	0x1fffdc,
	0x3fffd8,
	0x7fffe5,
	0x3fffd9,
	0x7fffe6,
	0x7fffe7,
	0xffffef,
	0x3fffda,
	0x1fffdd,
	0xfffe9,
	0x3fffdb,
	0x3fffdc,
	0x7fffe8,
	0x7fffe9,
	0x1fffde,
	0x7fffea,
	0x3fffdd,
	0x3fffde,
	0xfffff0,
	0x1fffdf,
	0x3fffdf,
	0x7fffeb,
	0x7fffec,
	0x1fffe0,
	0x1fffe1,
	0x3fffe0,
	0x1fffe2,
	0x7fffed,
	0x3fffe1,
	0x7fffee,
	0x7fffef,
	0xfffea,
	0x3fffe2,
	0x3fffe3,
	0x3fffe4,
	0x7ffff0,
	0x3fffe5,
	0x3fffe6,
	0x7ffff1,
	0x3ffffe0,
	0x3ffffe1,
	0xfffeb,
	0x7fff1,
	0x3fffe7,
	0x7ffff2,
	0x3fffe8,
	0x1ffffec,
	0x3ffffe2,
	0x3ffffe3,
	0x3ffffe4,
	0x7ffffde,
	0x7ffffdf,
	0x3ffffe5,
	0xfffff1,
	0x1ffffed,
	0x7fff2,
	0x1fffe3,
	0x3ffffe6,
	0x7ffffe0,
	0x7ffffe1,
	0x3ffffe7,
	0x7ffffe2,
	0xfffff2,
	0x1fffe4,
	0x1fffe5,
	0x3ffffe8,
	0x3ffffe9,
	0xffffffd,
	0x7ffffe3,
	0x7ffffe4,
	0x7ffffe5,
	0xfffec,
	0xfffff3,
	0xfffed,
	0x1fffe6,
	0x3fffe9,
	0x1fffe7,
	0x1fffe8,
	0x7ffff3,
	0x3fffea,
	0x3fffeb,
	0x1ffffee,
	0x1ffffef,
	0xfffff4,
	0xfffff5,
	0x3ffffea,
	0x7ffff4,
	0x3ffffeb,
	0x7ffffe6,
	0x3ffffec,
	0x3ffffed,
	0x7ffffe7,
	0x7ffffe8,
	0x7ffffe9,
	0x7ffffea,
	0x7ffffeb,
	0xffffffe,
	0x7ffffec,
	0x7ffffed,
	0x7ffffee,
	0x7ffffef,
	0x7fffff0,
	0x3ffffee,
}

var huffmanCodeLen = [256]uint8{
	13, 23, 28, 28, 28, 28, 28, 28, 28, 24, 30, 28, 28, 30, 28, 28,
	28, 28, 28, 28, 28, 28, 30, 28, 28, 28, 28, 28, 28, 28, 28, 28,
	6, 10, 10, 12, 13, 6, 8, 11, 10, 10, 8, 11, 8, 6, 6, 6,
	5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 15, 6, 12, 10,
	13, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 8, 7, 8, 13, 19, 13, 14, 6,
	15, 5, 6, 5, 6, 5, 6, 6, 6, 5, 7, 7, 6, 6, 6, 5,
	6, 7, 6, 5, 5, 6, 7, 7, 7, 7, 7, 15, 11, 14, 13, 28,
	20, 22, 20, 20, 22, 22, 22, 23, 22, 23, 23, 23, 23, 23, 24, 23,
	24, 24, 22, 23, 24, 23, 23, 23, 23, 21, 22, 23, 22, 23, 23, 24,
	22, 21, 20, 22, 22, 23, 23, 21, 23, 22, 22, 24, 21, 22, 23, 23,
	21, 21, 22, 21, 23, 22, 23, 23, 20, 22, 22, 22, 23, 22, 22, 23,
	26, 26, 20, 19, 22, 23, 22, 25, 26, 26, 26, 27, 27, 26, 24, 25,
	19, 21, 26, 27, 27, 26, 27, 24, 21, 21, 26, 26, 28, 27, 27, 27,
	20, 24, 20, 21, 22, 21, 21, 23, 22, 22, 25, 25, 24, 24, 26, 23,
	26, 27, 26, 26, 27, 27, 27, 27, 27, 28, 27, 27, 27, 27, 27, 26,
}
