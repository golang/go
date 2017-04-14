// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpack

import (
	"bufio"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func TestHeaderFieldTable(t *testing.T) {
	table := &headerFieldTable{}
	table.init()
	table.addEntry(pair("key1", "value1-1"))
	table.addEntry(pair("key2", "value2-1"))
	table.addEntry(pair("key1", "value1-2"))
	table.addEntry(pair("key3", "value3-1"))
	table.addEntry(pair("key4", "value4-1"))
	table.addEntry(pair("key2", "value2-2"))

	// Tests will be run twice: once before evicting anything, and
	// again after evicting the three oldest entries.
	tests := []struct {
		f                 HeaderField
		beforeWantStaticI uint64
		beforeWantMatch   bool
		afterWantStaticI  uint64
		afterWantMatch    bool
	}{
		{HeaderField{"key1", "value1-1", false}, 1, true, 0, false},
		{HeaderField{"key1", "value1-2", false}, 3, true, 0, false},
		{HeaderField{"key1", "value1-3", false}, 3, false, 0, false},
		{HeaderField{"key2", "value2-1", false}, 2, true, 3, false},
		{HeaderField{"key2", "value2-2", false}, 6, true, 3, true},
		{HeaderField{"key2", "value2-3", false}, 6, false, 3, false},
		{HeaderField{"key4", "value4-1", false}, 5, true, 2, true},
		// Name match only, because sensitive.
		{HeaderField{"key4", "value4-1", true}, 5, false, 2, false},
		// Key not found.
		{HeaderField{"key5", "value5-x", false}, 0, false, 0, false},
	}

	staticToDynamic := func(i uint64) uint64 {
		if i == 0 {
			return 0
		}
		return uint64(table.len()) - i + 1 // dynamic is the reversed table
	}

	searchStatic := func(f HeaderField) (uint64, bool) {
		old := staticTable
		staticTable = table
		defer func() { staticTable = old }()
		return staticTable.search(f)
	}

	searchDynamic := func(f HeaderField) (uint64, bool) {
		return table.search(f)
	}

	for _, test := range tests {
		gotI, gotMatch := searchStatic(test.f)
		if wantI, wantMatch := test.beforeWantStaticI, test.beforeWantMatch; gotI != wantI || gotMatch != wantMatch {
			t.Errorf("before evictions: searchStatic(%+v)=%v,%v want %v,%v", test.f, gotI, gotMatch, wantI, wantMatch)
		}
		gotI, gotMatch = searchDynamic(test.f)
		wantDynamicI := staticToDynamic(test.beforeWantStaticI)
		if wantI, wantMatch := wantDynamicI, test.beforeWantMatch; gotI != wantI || gotMatch != wantMatch {
			t.Errorf("before evictions: searchDynamic(%+v)=%v,%v want %v,%v", test.f, gotI, gotMatch, wantI, wantMatch)
		}
	}

	table.evictOldest(3)

	for _, test := range tests {
		gotI, gotMatch := searchStatic(test.f)
		if wantI, wantMatch := test.afterWantStaticI, test.afterWantMatch; gotI != wantI || gotMatch != wantMatch {
			t.Errorf("after evictions: searchStatic(%+v)=%v,%v want %v,%v", test.f, gotI, gotMatch, wantI, wantMatch)
		}
		gotI, gotMatch = searchDynamic(test.f)
		wantDynamicI := staticToDynamic(test.afterWantStaticI)
		if wantI, wantMatch := wantDynamicI, test.afterWantMatch; gotI != wantI || gotMatch != wantMatch {
			t.Errorf("after evictions: searchDynamic(%+v)=%v,%v want %v,%v", test.f, gotI, gotMatch, wantI, wantMatch)
		}
	}
}

func TestHeaderFieldTable_LookupMapEviction(t *testing.T) {
	table := &headerFieldTable{}
	table.init()
	table.addEntry(pair("key1", "value1-1"))
	table.addEntry(pair("key2", "value2-1"))
	table.addEntry(pair("key1", "value1-2"))
	table.addEntry(pair("key3", "value3-1"))
	table.addEntry(pair("key4", "value4-1"))
	table.addEntry(pair("key2", "value2-2"))

	// evict all pairs
	table.evictOldest(table.len())

	if l := table.len(); l > 0 {
		t.Errorf("table.len() = %d, want 0", l)
	}

	if l := len(table.byName); l > 0 {
		t.Errorf("len(table.byName) = %d, want 0", l)
	}

	if l := len(table.byNameValue); l > 0 {
		t.Errorf("len(table.byNameValue) = %d, want 0", l)
	}
}

func TestStaticTable(t *testing.T) {
	fromSpec := `
          +-------+-----------------------------+---------------+
          | 1     | :authority                  |               |
          | 2     | :method                     | GET           |
          | 3     | :method                     | POST          |
          | 4     | :path                       | /             |
          | 5     | :path                       | /index.html   |
          | 6     | :scheme                     | http          |
          | 7     | :scheme                     | https         |
          | 8     | :status                     | 200           |
          | 9     | :status                     | 204           |
          | 10    | :status                     | 206           |
          | 11    | :status                     | 304           |
          | 12    | :status                     | 400           |
          | 13    | :status                     | 404           |
          | 14    | :status                     | 500           |
          | 15    | accept-charset              |               |
          | 16    | accept-encoding             | gzip, deflate |
          | 17    | accept-language             |               |
          | 18    | accept-ranges               |               |
          | 19    | accept                      |               |
          | 20    | access-control-allow-origin |               |
          | 21    | age                         |               |
          | 22    | allow                       |               |
          | 23    | authorization               |               |
          | 24    | cache-control               |               |
          | 25    | content-disposition         |               |
          | 26    | content-encoding            |               |
          | 27    | content-language            |               |
          | 28    | content-length              |               |
          | 29    | content-location            |               |
          | 30    | content-range               |               |
          | 31    | content-type                |               |
          | 32    | cookie                      |               |
          | 33    | date                        |               |
          | 34    | etag                        |               |
          | 35    | expect                      |               |
          | 36    | expires                     |               |
          | 37    | from                        |               |
          | 38    | host                        |               |
          | 39    | if-match                    |               |
          | 40    | if-modified-since           |               |
          | 41    | if-none-match               |               |
          | 42    | if-range                    |               |
          | 43    | if-unmodified-since         |               |
          | 44    | last-modified               |               |
          | 45    | link                        |               |
          | 46    | location                    |               |
          | 47    | max-forwards                |               |
          | 48    | proxy-authenticate          |               |
          | 49    | proxy-authorization         |               |
          | 50    | range                       |               |
          | 51    | referer                     |               |
          | 52    | refresh                     |               |
          | 53    | retry-after                 |               |
          | 54    | server                      |               |
          | 55    | set-cookie                  |               |
          | 56    | strict-transport-security   |               |
          | 57    | transfer-encoding           |               |
          | 58    | user-agent                  |               |
          | 59    | vary                        |               |
          | 60    | via                         |               |
          | 61    | www-authenticate            |               |
          +-------+-----------------------------+---------------+
`
	bs := bufio.NewScanner(strings.NewReader(fromSpec))
	re := regexp.MustCompile(`\| (\d+)\s+\| (\S+)\s*\| (\S(.*\S)?)?\s+\|`)
	for bs.Scan() {
		l := bs.Text()
		if !strings.Contains(l, "|") {
			continue
		}
		m := re.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		i, err := strconv.Atoi(m[1])
		if err != nil {
			t.Errorf("Bogus integer on line %q", l)
			continue
		}
		if i < 1 || i > staticTable.len() {
			t.Errorf("Bogus index %d on line %q", i, l)
			continue
		}
		if got, want := staticTable.ents[i-1].Name, m[2]; got != want {
			t.Errorf("header index %d name = %q; want %q", i, got, want)
		}
		if got, want := staticTable.ents[i-1].Value, m[3]; got != want {
			t.Errorf("header index %d value = %q; want %q", i, got, want)
		}
	}
	if err := bs.Err(); err != nil {
		t.Error(err)
	}
}
