// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package suffixarray

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

type testCase struct {
	name     string   // name of test case
	source   string   // source to index
	patterns []string // patterns to lookup
}

var testCases = []testCase{
	{
		"empty string",
		"",
		[]string{
			"",
			"foo",
			"(foo)",
			".*",
			"a*",
		},
	},

	{
		"all a's",
		"aaaaaaaaaa", // 10 a's
		[]string{
			"",
			"a",
			"aa",
			"aaa",
			"aaaa",
			"aaaaa",
			"aaaaaa",
			"aaaaaaa",
			"aaaaaaaa",
			"aaaaaaaaa",
			"aaaaaaaaaa",
			"aaaaaaaaaaa", // 11 a's
			".",
			".*",
			"a+",
			"aa+",
			"aaaa[b]?",
			"aaa*",
		},
	},

	{
		"abc",
		"abc",
		[]string{
			"a",
			"b",
			"c",
			"ab",
			"bc",
			"abc",
			"a.c",
			"a(b|c)",
			"abc?",
		},
	},

	{
		"barbara*3",
		"barbarabarbarabarbara",
		[]string{
			"a",
			"bar",
			"rab",
			"arab",
			"barbar",
			"bara?bar",
		},
	},

	{
		"typing drill",
		"Now is the time for all good men to come to the aid of their country.",
		[]string{
			"Now",
			"the time",
			"to come the aid",
			"is the time for all good men to come to the aid of their",
			"to (come|the)?",
		},
	},

	{
		"godoc simulation",
		"package main\n\nimport(\n    \"rand\"\n    ",
		[]string{},
	},
}

// find all occurrences of s in source; report at most n occurrences
func find(src, s string, n int) []int {
	var res []int
	if s != "" && n != 0 {
		// find at most n occurrences of s in src
		for i := -1; n < 0 || len(res) < n; {
			j := strings.Index(src[i+1:], s)
			if j < 0 {
				break
			}
			i += j + 1
			res = append(res, i)
		}
	}
	return res
}

func testLookup(t *testing.T, tc *testCase, x *Index, s string, n int) {
	res := x.Lookup([]byte(s), n)
	exp := find(tc.source, s, n)

	// check that the lengths match
	if len(res) != len(exp) {
		t.Errorf("test %q, lookup %q (n = %d): expected %d results; got %d", tc.name, s, n, len(exp), len(res))
	}

	// if n >= 0 the number of results is limited --- unless n >= all results,
	// we may obtain different positions from the Index and from find (because
	// Index may not find the results in the same order as find) => in general
	// we cannot simply check that the res and exp lists are equal

	// check that each result is in fact a correct match and there are no duplicates
	sort.Ints(res)
	for i, r := range res {
		if r < 0 || len(tc.source) <= r {
			t.Errorf("test %q, lookup %q, result %d (n = %d): index %d out of range [0, %d[", tc.name, s, i, n, r, len(tc.source))
		} else if !strings.HasPrefix(tc.source[r:], s) {
			t.Errorf("test %q, lookup %q, result %d (n = %d): index %d not a match", tc.name, s, i, n, r)
		}
		if i > 0 && res[i-1] == r {
			t.Errorf("test %q, lookup %q, result %d (n = %d): found duplicate index %d", tc.name, s, i, n, r)
		}
	}

	if n < 0 {
		// all results computed - sorted res and exp must be equal
		for i, r := range res {
			e := exp[i]
			if r != e {
				t.Errorf("test %q, lookup %q, result %d: expected index %d; got %d", tc.name, s, i, e, r)
			}
		}
	}
}

func testFindAllIndex(t *testing.T, tc *testCase, x *Index, rx *regexp.Regexp, n int) {
	res := x.FindAllIndex(rx, n)
	exp := rx.FindAllStringIndex(tc.source, n)

	// check that the lengths match
	if len(res) != len(exp) {
		t.Errorf("test %q, FindAllIndex %q (n = %d): expected %d results; got %d", tc.name, rx, n, len(exp), len(res))
	}

	// if n >= 0 the number of results is limited --- unless n >= all results,
	// we may obtain different positions from the Index and from regexp (because
	// Index may not find the results in the same order as regexp) => in general
	// we cannot simply check that the res and exp lists are equal

	// check that each result is in fact a correct match and the result is sorted
	for i, r := range res {
		if r[0] < 0 || r[0] > r[1] || len(tc.source) < r[1] {
			t.Errorf("test %q, FindAllIndex %q, result %d (n == %d): illegal match [%d, %d]", tc.name, rx, i, n, r[0], r[1])
		} else if !rx.MatchString(tc.source[r[0]:r[1]]) {
			t.Errorf("test %q, FindAllIndex %q, result %d (n = %d): [%d, %d] not a match", tc.name, rx, i, n, r[0], r[1])
		}
	}

	if n < 0 {
		// all results computed - sorted res and exp must be equal
		for i, r := range res {
			e := exp[i]
			if r[0] != e[0] || r[1] != e[1] {
				t.Errorf("test %q, FindAllIndex %q, result %d: expected match [%d, %d]; got [%d, %d]",
					tc.name, rx, i, e[0], e[1], r[0], r[1])
			}
		}
	}
}

func testLookups(t *testing.T, tc *testCase, x *Index, n int) {
	for _, pat := range tc.patterns {
		testLookup(t, tc, x, pat, n)
		if rx, err := regexp.Compile(pat); err == nil {
			testFindAllIndex(t, tc, x, rx, n)
		}
	}
}

// index is used to hide the sort.Interface
type index Index

func (x *index) Len() int           { return x.sa.len() }
func (x *index) Less(i, j int) bool { return bytes.Compare(x.at(i), x.at(j)) < 0 }
func (x *index) Swap(i, j int) {
	if x.sa.int32 != nil {
		x.sa.int32[i], x.sa.int32[j] = x.sa.int32[j], x.sa.int32[i]
	} else {
		x.sa.int64[i], x.sa.int64[j] = x.sa.int64[j], x.sa.int64[i]
	}
}

func (x *index) at(i int) []byte {
	return x.data[x.sa.get(i):]
}

func testConstruction(t *testing.T, tc *testCase, x *Index) {
	if !sort.IsSorted((*index)(x)) {
		t.Errorf("failed testConstruction %s", tc.name)
	}
}

func equal(x, y *Index) bool {
	if !bytes.Equal(x.data, y.data) {
		return false
	}
	if x.sa.len() != y.sa.len() {
		return false
	}
	n := x.sa.len()
	for i := 0; i < n; i++ {
		if x.sa.get(i) != y.sa.get(i) {
			return false
		}
	}
	return true
}

// returns the serialized index size
func testSaveRestore(t *testing.T, tc *testCase, x *Index) int {
	var buf bytes.Buffer
	if err := x.Write(&buf); err != nil {
		t.Errorf("failed writing index %s (%s)", tc.name, err)
	}
	size := buf.Len()
	var y Index
	if err := y.Read(bytes.NewReader(buf.Bytes())); err != nil {
		t.Errorf("failed reading index %s (%s)", tc.name, err)
	}
	if !equal(x, &y) {
		t.Errorf("restored index doesn't match saved index %s", tc.name)
	}

	old := maxData32
	defer func() {
		maxData32 = old
	}()
	// Reread as forced 32.
	y = Index{}
	maxData32 = realMaxData32
	if err := y.Read(bytes.NewReader(buf.Bytes())); err != nil {
		t.Errorf("failed reading index %s (%s)", tc.name, err)
	}
	if !equal(x, &y) {
		t.Errorf("restored index doesn't match saved index %s", tc.name)
	}

	// Reread as forced 64.
	y = Index{}
	maxData32 = -1
	if err := y.Read(bytes.NewReader(buf.Bytes())); err != nil {
		t.Errorf("failed reading index %s (%s)", tc.name, err)
	}
	if !equal(x, &y) {
		t.Errorf("restored index doesn't match saved index %s", tc.name)
	}

	return size
}

func testIndex(t *testing.T) {
	for _, tc := range testCases {
		x := New([]byte(tc.source))
		testConstruction(t, &tc, x)
		testSaveRestore(t, &tc, x)
		testLookups(t, &tc, x, 0)
		testLookups(t, &tc, x, 1)
		testLookups(t, &tc, x, 10)
		testLookups(t, &tc, x, 2e9)
		testLookups(t, &tc, x, -1)
	}
}

func TestIndex32(t *testing.T) {
	testIndex(t)
}

func TestIndex64(t *testing.T) {
	maxData32 = -1
	defer func() {
		maxData32 = realMaxData32
	}()
	testIndex(t)
}

var (
	benchdata = make([]byte, 1e6)
	benchrand = make([]byte, 1e6)
)

// Of all possible inputs, the random bytes have the least amount of substring
// repetition, and the repeated bytes have the most. For most algorithms,
// the running time of every input will be between these two.
func benchmarkNew(b *testing.B, random bool) {
	b.ReportAllocs()
	b.StopTimer()
	data := benchdata
	if random {
		data = benchrand
		if data[0] == 0 {
			for i := range data {
				data[i] = byte(rand.Intn(256))
			}
		}
	}
	b.StartTimer()
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		New(data)
	}
}

func makeText(name string) ([]byte, error) {
	var data []byte
	switch name {
	case "opticks":
		var err error
		data, err = ioutil.ReadFile("../../testdata/Isaac.Newton-Opticks.txt")
		if err != nil {
			return nil, err
		}
	case "go":
		err := filepath.Walk("../..", func(path string, info os.FileInfo, err error) error {
			if err == nil && strings.HasSuffix(path, ".go") && !info.IsDir() {
				file, err := ioutil.ReadFile(path)
				if err != nil {
					return err
				}
				data = append(data, file...)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	case "zero":
		data = make([]byte, 50e6)
	case "rand":
		data = make([]byte, 50e6)
		for i := range data {
			data[i] = byte(rand.Intn(256))
		}
	}
	return data, nil
}

func setBits(bits int) (cleanup func()) {
	if bits == 32 {
		maxData32 = realMaxData32
	} else {
		maxData32 = -1 // force use of 64-bit code
	}
	return func() {
		maxData32 = realMaxData32
	}
}

func BenchmarkNew(b *testing.B) {
	for _, text := range []string{"opticks", "go", "zero", "rand"} {
		b.Run("text="+text, func(b *testing.B) {
			data, err := makeText(text)
			if err != nil {
				b.Fatal(err)
			}
			if testing.Short() && len(data) > 5e6 {
				data = data[:5e6]
			}
			for _, size := range []int{100e3, 500e3, 1e6, 5e6, 10e6, 50e6} {
				if len(data) < size {
					continue
				}
				data := data[:size]
				name := fmt.Sprintf("%dK", size/1e3)
				if size >= 1e6 {
					name = fmt.Sprintf("%dM", size/1e6)
				}
				b.Run("size="+name, func(b *testing.B) {
					for _, bits := range []int{32, 64} {
						if ^uint(0) == 0xffffffff && bits == 64 {
							continue
						}
						b.Run(fmt.Sprintf("bits=%d", bits), func(b *testing.B) {
							cleanup := setBits(bits)
							defer cleanup()

							b.SetBytes(int64(len(data)))
							b.ReportAllocs()
							for i := 0; i < b.N; i++ {
								New(data)
							}
						})
					}
				})
			}
		})
	}
}

func BenchmarkSaveRestore(b *testing.B) {
	r := rand.New(rand.NewSource(0x5a77a1)) // guarantee always same sequence
	data := make([]byte, 1<<20)             // 1MB of data to index
	for i := range data {
		data[i] = byte(r.Intn(256))
	}
	for _, bits := range []int{32, 64} {
		if ^uint(0) == 0xffffffff && bits == 64 {
			continue
		}
		b.Run(fmt.Sprintf("bits=%d", bits), func(b *testing.B) {
			cleanup := setBits(bits)
			defer cleanup()

			b.StopTimer()
			x := New(data)
			size := testSaveRestore(nil, nil, x)       // verify correctness
			buf := bytes.NewBuffer(make([]byte, size)) // avoid growing
			b.SetBytes(int64(size))
			b.StartTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				buf.Reset()
				if err := x.Write(buf); err != nil {
					b.Fatal(err)
				}
				var y Index
				if err := y.Read(buf); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
