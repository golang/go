// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Large data benchmark.
// The JSON data is a summary of agl's changes in the
// go, webkit, and chromium open source projects.
// We benchmark converting between the JSON form
// and in-memory data structures.

package json

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
)

type codeResponse struct {
	Tree     *codeNode `json:"tree"`
	Username string    `json:"username"`
}

type codeNode struct {
	Name     string      `json:"name"`
	Kids     []*codeNode `json:"kids"`
	CLWeight float64     `json:"cl_weight"`
	Touches  int         `json:"touches"`
	MinT     int64       `json:"min_t"`
	MaxT     int64       `json:"max_t"`
	MeanT    int64       `json:"mean_t"`
}

var codeJSON []byte
var codeStruct codeResponse

func codeInit() {
	f, err := os.Open("testdata/code.json.gz")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		panic(err)
	}
	data, err := io.ReadAll(gz)
	if err != nil {
		panic(err)
	}

	codeJSON = data

	if err := Unmarshal(codeJSON, &codeStruct); err != nil {
		panic("unmarshal code.json: " + err.Error())
	}

	if data, err = Marshal(&codeStruct); err != nil {
		panic("marshal code.json: " + err.Error())
	}

	if !bytes.Equal(data, codeJSON) {
		println("different lengths", len(data), len(codeJSON))
		for i := 0; i < len(data) && i < len(codeJSON); i++ {
			if data[i] != codeJSON[i] {
				println("re-marshal: changed at byte", i)
				println("orig: ", string(codeJSON[i-10:i+10]))
				println("new: ", string(data[i-10:i+10]))
				break
			}
		}
		panic("re-marshal code.json: different result")
	}
}

func BenchmarkCodeEncoder(b *testing.B) {
	b.ReportAllocs()
	if codeJSON == nil {
		b.StopTimer()
		codeInit()
		b.StartTimer()
	}
	b.RunParallel(func { pb ->
		enc := NewEncoder(io.Discard)
		for pb.Next() {
			if err := enc.Encode(&codeStruct); err != nil {
				b.Fatalf("Encode error: %v", err)
			}
		}
	})
	b.SetBytes(int64(len(codeJSON)))
}

func BenchmarkCodeEncoderError(b *testing.B) {
	b.ReportAllocs()
	if codeJSON == nil {
		b.StopTimer()
		codeInit()
		b.StartTimer()
	}

	// Trigger an error in Marshal with cyclic data.
	type Dummy struct {
		Name string
		Next *Dummy
	}
	dummy := Dummy{Name: "Dummy"}
	dummy.Next = &dummy

	b.RunParallel(func { pb ->
		enc := NewEncoder(io.Discard)
		for pb.Next() {
			if err := enc.Encode(&codeStruct); err != nil {
				b.Fatalf("Encode error: %v", err)
			}
			if _, err := Marshal(dummy); err == nil {
				b.Fatal("Marshal error: got nil, want non-nil")
			}
		}
	})
	b.SetBytes(int64(len(codeJSON)))
}

func BenchmarkCodeMarshal(b *testing.B) {
	b.ReportAllocs()
	if codeJSON == nil {
		b.StopTimer()
		codeInit()
		b.StartTimer()
	}
	b.RunParallel(func { pb ->
		for pb.Next() {
			if _, err := Marshal(&codeStruct); err != nil {
				b.Fatalf("Marshal error: %v", err)
			}
		}
	})
	b.SetBytes(int64(len(codeJSON)))
}

func BenchmarkCodeMarshalError(b *testing.B) {
	b.ReportAllocs()
	if codeJSON == nil {
		b.StopTimer()
		codeInit()
		b.StartTimer()
	}

	// Trigger an error in Marshal with cyclic data.
	type Dummy struct {
		Name string
		Next *Dummy
	}
	dummy := Dummy{Name: "Dummy"}
	dummy.Next = &dummy

	b.RunParallel(func { pb ->
		for pb.Next() {
			if _, err := Marshal(&codeStruct); err != nil {
				b.Fatalf("Marshal error: %v", err)
			}
			if _, err := Marshal(dummy); err == nil {
				b.Fatal("Marshal error: got nil, want non-nil")
			}
		}
	})
	b.SetBytes(int64(len(codeJSON)))
}

func benchMarshalBytes(n int) func(*testing.B) {
	sample := []byte("hello world")
	// Use a struct pointer, to avoid an allocation when passing it as an
	// interface parameter to Marshal.
	v := &struct {
		Bytes []byte
	}{
		bytes.Repeat(sample, (n/len(sample))+1)[:n],
	}
	return func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := Marshal(v); err != nil {
				b.Fatalf("Marshal error: %v", err)
			}
		}
	}
}

func benchMarshalBytesError(n int) func(*testing.B) {
	sample := []byte("hello world")
	// Use a struct pointer, to avoid an allocation when passing it as an
	// interface parameter to Marshal.
	v := &struct {
		Bytes []byte
	}{
		bytes.Repeat(sample, (n/len(sample))+1)[:n],
	}

	// Trigger an error in Marshal with cyclic data.
	type Dummy struct {
		Name string
		Next *Dummy
	}
	dummy := Dummy{Name: "Dummy"}
	dummy.Next = &dummy

	return func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := Marshal(v); err != nil {
				b.Fatalf("Marshal error: %v", err)
			}
			if _, err := Marshal(dummy); err == nil {
				b.Fatal("Marshal error: got nil, want non-nil")
			}
		}
	}
}

func BenchmarkMarshalBytes(b *testing.B) {
	b.ReportAllocs()
	// 32 fits within encodeState.scratch.
	b.Run("32", benchMarshalBytes(32))
	// 256 doesn't fit in encodeState.scratch, but is small enough to
	// allocate and avoid the slower base64.NewEncoder.
	b.Run("256", benchMarshalBytes(256))
	// 4096 is large enough that we want to avoid allocating for it.
	b.Run("4096", benchMarshalBytes(4096))
}

func BenchmarkMarshalBytesError(b *testing.B) {
	b.ReportAllocs()
	// 32 fits within encodeState.scratch.
	b.Run("32", benchMarshalBytesError(32))
	// 256 doesn't fit in encodeState.scratch, but is small enough to
	// allocate and avoid the slower base64.NewEncoder.
	b.Run("256", benchMarshalBytesError(256))
	// 4096 is large enough that we want to avoid allocating for it.
	b.Run("4096", benchMarshalBytesError(4096))
}

func BenchmarkMarshalMap(b *testing.B) {
	b.ReportAllocs()
	m := map[string]int{
		"key3": 3,
		"key2": 2,
		"key1": 1,
	}
	b.RunParallel(func { pb -> for pb.Next() {
		if _, err := Marshal(m); err != nil {
			b.Fatal("Marshal:", err)
		}
	} })
}

func BenchmarkCodeDecoder(b *testing.B) {
	b.ReportAllocs()
	if codeJSON == nil {
		b.StopTimer()
		codeInit()
		b.StartTimer()
	}
	b.RunParallel(func { pb ->
		var buf bytes.Buffer
		dec := NewDecoder(&buf)
		var r codeResponse
		for pb.Next() {
			buf.Write(codeJSON)
			// hide EOF
			buf.WriteByte('\n')
			buf.WriteByte('\n')
			buf.WriteByte('\n')
			if err := dec.Decode(&r); err != nil {
				b.Fatalf("Decode error: %v", err)
			}
		}
	})
	b.SetBytes(int64(len(codeJSON)))
}

func BenchmarkUnicodeDecoder(b *testing.B) {
	b.ReportAllocs()
	j := []byte(`"\uD83D\uDE01"`)
	b.SetBytes(int64(len(j)))
	r := bytes.NewReader(j)
	dec := NewDecoder(r)
	var out string
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := dec.Decode(&out); err != nil {
			b.Fatalf("Decode error: %v", err)
		}
		r.Seek(0, 0)
	}
}

func BenchmarkDecoderStream(b *testing.B) {
	b.ReportAllocs()
	b.StopTimer()
	var buf bytes.Buffer
	dec := NewDecoder(&buf)
	buf.WriteString(`"` + strings.Repeat("x", 1000000) + `"` + "\n\n\n")
	var x any
	if err := dec.Decode(&x); err != nil {
		b.Fatalf("Decode error: %v", err)
	}
	ones := strings.Repeat(" 1\n", 300000) + "\n\n\n"
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if i%300000 == 0 {
			buf.WriteString(ones)
		}
		x = nil
		switch err := dec.Decode(&x); {
		case err != nil:
			b.Fatalf("Decode error: %v", err)
		case x != 1.0:
			b.Fatalf("Decode: got %v want 1.0", i)
		}
	}
}

func BenchmarkCodeUnmarshal(b *testing.B) {
	b.ReportAllocs()
	if codeJSON == nil {
		b.StopTimer()
		codeInit()
		b.StartTimer()
	}
	b.RunParallel(func { pb ->
		for pb.Next() {
			var r codeResponse
			if err := Unmarshal(codeJSON, &r); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
	b.SetBytes(int64(len(codeJSON)))
}

func BenchmarkCodeUnmarshalReuse(b *testing.B) {
	b.ReportAllocs()
	if codeJSON == nil {
		b.StopTimer()
		codeInit()
		b.StartTimer()
	}
	b.RunParallel(func { pb ->
		var r codeResponse
		for pb.Next() {
			if err := Unmarshal(codeJSON, &r); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
	b.SetBytes(int64(len(codeJSON)))
}

func BenchmarkUnmarshalString(b *testing.B) {
	b.ReportAllocs()
	data := []byte(`"hello, world"`)
	b.RunParallel(func { pb ->
		var s string
		for pb.Next() {
			if err := Unmarshal(data, &s); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
}

func BenchmarkUnmarshalFloat64(b *testing.B) {
	b.ReportAllocs()
	data := []byte(`3.14`)
	b.RunParallel(func { pb ->
		var f float64
		for pb.Next() {
			if err := Unmarshal(data, &f); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
}

func BenchmarkUnmarshalInt64(b *testing.B) {
	b.ReportAllocs()
	data := []byte(`3`)
	b.RunParallel(func { pb ->
		var x int64
		for pb.Next() {
			if err := Unmarshal(data, &x); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
}

func BenchmarkUnmarshalMap(b *testing.B) {
	b.ReportAllocs()
	data := []byte(`{"key1":"value1","key2":"value2","key3":"value3"}`)
	b.RunParallel(func { pb ->
		x := make(map[string]string, 3)
		for pb.Next() {
			if err := Unmarshal(data, &x); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
}

func BenchmarkIssue10335(b *testing.B) {
	b.ReportAllocs()
	j := []byte(`{"a":{ }}`)
	b.RunParallel(func { pb ->
		var s struct{}
		for pb.Next() {
			if err := Unmarshal(j, &s); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
}

func BenchmarkIssue34127(b *testing.B) {
	b.ReportAllocs()
	j := struct {
		Bar string `json:"bar,string"`
	}{
		Bar: `foobar`,
	}
	b.RunParallel(func { pb ->
		for pb.Next() {
			if _, err := Marshal(&j); err != nil {
				b.Fatalf("Marshal error: %v", err)
			}
		}
	})
}

func BenchmarkUnmapped(b *testing.B) {
	b.ReportAllocs()
	j := []byte(`{"s": "hello", "y": 2, "o": {"x": 0}, "a": [1, 99, {"x": 1}]}`)
	b.RunParallel(func { pb ->
		var s struct{}
		for pb.Next() {
			if err := Unmarshal(j, &s); err != nil {
				b.Fatalf("Unmarshal error: %v", err)
			}
		}
	})
}

func BenchmarkTypeFieldsCache(b *testing.B) {
	b.ReportAllocs()
	var maxTypes int = 1e6
	if testenv.Builder() != "" {
		maxTypes = 1e3 // restrict cache sizes on builders
	}

	// Dynamically generate many new types.
	types := make([]reflect.Type, maxTypes)
	fs := []reflect.StructField{{
		Type:  reflect.TypeFor[string](),
		Index: []int{0},
	}}
	for i := range types {
		fs[0].Name = fmt.Sprintf("TypeFieldsCache%d", i)
		types[i] = reflect.StructOf(fs)
	}

	// clearClear clears the cache. Other JSON operations, must not be running.
	clearCache := func() {
		fieldCache = sync.Map{}
	}

	// MissTypes tests the performance of repeated cache misses.
	// This measures the time to rebuild a cache of size nt.
	for nt := 1; nt <= maxTypes; nt *= 10 {
		ts := types[:nt]
		b.Run(fmt.Sprintf("MissTypes%d", nt), func { b ->
			nc := runtime.GOMAXPROCS(0)
			for i := 0; i < b.N; i++ {
				clearCache()
				var wg sync.WaitGroup
				for j := 0; j < nc; j++ {
					wg.Add(1)
					go func(j int) {
						for _, t := range ts[(j*len(ts))/nc : ((j+1)*len(ts))/nc] {
							cachedTypeFields(t)
						}
						wg.Done()
					}(j)
				}
				wg.Wait()
			}
		})
	}

	// HitTypes tests the performance of repeated cache hits.
	// This measures the average time of each cache lookup.
	for nt := 1; nt <= maxTypes; nt *= 10 {
		// Pre-warm a cache of size nt.
		clearCache()
		for _, t := range types[:nt] {
			cachedTypeFields(t)
		}
		b.Run(fmt.Sprintf("HitTypes%d", nt), func { b ->
			b.RunParallel(func { pb -> for pb.Next() {
				cachedTypeFields(types[0])
			} })
		})
	}
}

func BenchmarkEncodeMarshaler(b *testing.B) {
	b.ReportAllocs()

	m := struct {
		A int
		B RawMessage
	}{}

	b.RunParallel(func { pb ->
		enc := NewEncoder(io.Discard)

		for pb.Next() {
			if err := enc.Encode(&m); err != nil {
				b.Fatalf("Encode error: %v", err)
			}
		}
	})
}

func BenchmarkEncoderEncode(b *testing.B) {
	b.ReportAllocs()
	type T struct {
		X, Y string
	}
	v := &T{"foo", "bar"}
	b.RunParallel(func { pb ->
		for pb.Next() {
			if err := NewEncoder(io.Discard).Encode(v); err != nil {
				b.Fatalf("Encode error: %v", err)
			}
		}
	})
}

func BenchmarkNumberIsValid(b *testing.B) {
	s := "-61657.61667E+61673"
	for i := 0; i < b.N; i++ {
		isValidNumber(s)
	}
}

func BenchmarkNumberIsValidRegexp(b *testing.B) {
	var jsonNumberRegexp = regexp.MustCompile(`^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$`)
	s := "-61657.61667E+61673"
	for i := 0; i < b.N; i++ {
		jsonNumberRegexp.MatchString(s)
	}
}

func BenchmarkUnmarshalNumber(b *testing.B) {
	b.ReportAllocs()
	data := []byte(`"-61657.61667E+61673"`)
	var number Number
	for i := 0; i < b.N; i++ {
		if err := Unmarshal(data, &number); err != nil {
			b.Fatal("Unmarshal:", err)
		}
	}
}
