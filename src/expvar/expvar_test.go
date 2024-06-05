// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package expvar

import (
	"bytes"
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"net"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
)

// RemoveAll removes all exported variables.
// This is for tests only.
func RemoveAll() {
	vars.keysMu.Lock()
	defer vars.keysMu.Unlock()
	for _, k := range vars.keys {
		vars.m.Delete(k)
	}
	vars.keys = nil
}

func TestNil(t *testing.T) {
	RemoveAll()
	val := Get("missing")
	if val != nil {
		t.Errorf("got %v, want nil", val)
	}
}

func TestInt(t *testing.T) {
	RemoveAll()
	reqs := NewInt("requests")
	if i := reqs.Value(); i != 0 {
		t.Errorf("reqs.Value() = %v, want 0", i)
	}
	if reqs != Get("requests").(*Int) {
		t.Errorf("Get() failed.")
	}

	reqs.Add(1)
	reqs.Add(3)
	if i := reqs.Value(); i != 4 {
		t.Errorf("reqs.Value() = %v, want 4", i)
	}

	if s := reqs.String(); s != "4" {
		t.Errorf("reqs.String() = %q, want \"4\"", s)
	}

	reqs.Set(-2)
	if i := reqs.Value(); i != -2 {
		t.Errorf("reqs.Value() = %v, want -2", i)
	}
}

func BenchmarkIntAdd(b *testing.B) {
	var v Int

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v.Add(1)
		}
	})
}

func BenchmarkIntSet(b *testing.B) {
	var v Int

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v.Set(1)
		}
	})
}

func TestFloat(t *testing.T) {
	RemoveAll()
	reqs := NewFloat("requests-float")
	if reqs.f.Load() != 0.0 {
		t.Errorf("reqs.f = %v, want 0", reqs.f.Load())
	}
	if reqs != Get("requests-float").(*Float) {
		t.Errorf("Get() failed.")
	}

	reqs.Add(1.5)
	reqs.Add(1.25)
	if v := reqs.Value(); v != 2.75 {
		t.Errorf("reqs.Value() = %v, want 2.75", v)
	}

	if s := reqs.String(); s != "2.75" {
		t.Errorf("reqs.String() = %q, want \"4.64\"", s)
	}

	reqs.Add(-2)
	if v := reqs.Value(); v != 0.75 {
		t.Errorf("reqs.Value() = %v, want 0.75", v)
	}
}

func BenchmarkFloatAdd(b *testing.B) {
	var f Float

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			f.Add(1.0)
		}
	})
}

func BenchmarkFloatSet(b *testing.B) {
	var f Float

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			f.Set(1.0)
		}
	})
}

func TestString(t *testing.T) {
	RemoveAll()
	name := NewString("my-name")
	if s := name.Value(); s != "" {
		t.Errorf(`NewString("my-name").Value() = %q, want ""`, s)
	}

	name.Set("Mike")
	if s, want := name.String(), `"Mike"`; s != want {
		t.Errorf(`after name.Set("Mike"), name.String() = %q, want %q`, s, want)
	}
	if s, want := name.Value(), "Mike"; s != want {
		t.Errorf(`after name.Set("Mike"), name.Value() = %q, want %q`, s, want)
	}

	// Make sure we produce safe JSON output.
	name.Set("<")
	if s, want := name.String(), "\"\\u003c\""; s != want {
		t.Errorf(`after name.Set("<"), name.String() = %q, want %q`, s, want)
	}
}

func BenchmarkStringSet(b *testing.B) {
	var s String

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			s.Set("red")
		}
	})
}

func TestMapInit(t *testing.T) {
	RemoveAll()
	colors := NewMap("bike-shed-colors")
	colors.Add("red", 1)
	colors.Add("blue", 1)
	colors.Add("chartreuse", 1)

	n := 0
	colors.Do(func(KeyValue) { n++ })
	if n != 3 {
		t.Errorf("after three Add calls with distinct keys, Do should invoke f 3 times; got %v", n)
	}

	colors.Init()

	n = 0
	colors.Do(func(KeyValue) { n++ })
	if n != 0 {
		t.Errorf("after Init, Do should invoke f 0 times; got %v", n)
	}
}

func TestMapDelete(t *testing.T) {
	RemoveAll()
	colors := NewMap("bike-shed-colors")

	colors.Add("red", 1)
	colors.Add("red", 2)
	colors.Add("blue", 4)

	n := 0
	colors.Do(func(KeyValue) { n++ })
	if n != 2 {
		t.Errorf("after two Add calls with distinct keys, Do should invoke f 2 times; got %v", n)
	}

	colors.Delete("red")
	if v := colors.Get("red"); v != nil {
		t.Errorf("removed red, Get should return nil; got %v", v)
	}
	n = 0
	colors.Do(func(KeyValue) { n++ })
	if n != 1 {
		t.Errorf("removed red, Do should invoke f 1 times; got %v", n)
	}

	colors.Delete("notfound")
	n = 0
	colors.Do(func(KeyValue) { n++ })
	if n != 1 {
		t.Errorf("attempted to remove notfound, Do should invoke f 1 times; got %v", n)
	}

	colors.Delete("blue")
	colors.Delete("blue")
	if v := colors.Get("blue"); v != nil {
		t.Errorf("removed blue, Get should return nil; got %v", v)
	}
	n = 0
	colors.Do(func(KeyValue) { n++ })
	if n != 0 {
		t.Errorf("all keys removed, Do should invoke f 0 times; got %v", n)
	}
}

func TestMapCounter(t *testing.T) {
	RemoveAll()
	colors := NewMap("bike-shed-colors")

	colors.Add("red", 1)
	colors.Add("red", 2)
	colors.Add("blue", 4)
	colors.AddFloat(`green "midori"`, 4.125)
	if x := colors.Get("red").(*Int).Value(); x != 3 {
		t.Errorf("colors.m[\"red\"] = %v, want 3", x)
	}
	if x := colors.Get("blue").(*Int).Value(); x != 4 {
		t.Errorf("colors.m[\"blue\"] = %v, want 4", x)
	}
	if x := colors.Get(`green "midori"`).(*Float).Value(); x != 4.125 {
		t.Errorf("colors.m[`green \"midori\"] = %v, want 4.125", x)
	}

	// colors.String() should be '{"red":3, "blue":4}',
	// though the order of red and blue could vary.
	s := colors.String()
	var j any
	err := json.Unmarshal([]byte(s), &j)
	if err != nil {
		t.Errorf("colors.String() isn't valid JSON: %v", err)
	}
	m, ok := j.(map[string]any)
	if !ok {
		t.Error("colors.String() didn't produce a map.")
	}
	red := m["red"]
	x, ok := red.(float64)
	if !ok {
		t.Error("red.Kind() is not a number.")
	}
	if x != 3 {
		t.Errorf("red = %v, want 3", x)
	}
}

func TestMapNil(t *testing.T) {
	RemoveAll()
	const key = "key"
	m := NewMap("issue527719")
	m.Set(key, nil)
	s := m.String()
	var j any
	if err := json.Unmarshal([]byte(s), &j); err != nil {
		t.Fatalf("m.String() == %q isn't valid JSON: %v", s, err)
	}
	m2, ok := j.(map[string]any)
	if !ok {
		t.Fatalf("m.String() produced %T, wanted a map", j)
	}
	v, ok := m2[key]
	if !ok {
		t.Fatalf("missing %q in %v", key, m2)
	}
	if v != nil {
		t.Fatalf("m[%q] = %v, want nil", key, v)
	}
}

func BenchmarkMapSet(b *testing.B) {
	m := new(Map).Init()

	v := new(Int)

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m.Set("red", v)
		}
	})
}

func BenchmarkMapSetDifferent(b *testing.B) {
	procKeys := make([][]string, runtime.GOMAXPROCS(0))
	for i := range procKeys {
		keys := make([]string, 4)
		for j := range keys {
			keys[j] = fmt.Sprint(i, j)
		}
		procKeys[i] = keys
	}

	m := new(Map).Init()
	v := new(Int)
	b.ResetTimer()

	var n int32
	b.RunParallel(func(pb *testing.PB) {
		i := int(atomic.AddInt32(&n, 1)-1) % len(procKeys)
		keys := procKeys[i]

		for pb.Next() {
			for _, k := range keys {
				m.Set(k, v)
			}
		}
	})
}

// BenchmarkMapSetDifferentRandom simulates such a case where the concerned
// keys of Map.Set are generated dynamically and as a result insertion is
// out of order and the number of the keys may be large.
func BenchmarkMapSetDifferentRandom(b *testing.B) {
	keys := make([]string, 100)
	for i := range keys {
		keys[i] = fmt.Sprintf("%x", sha1.Sum([]byte(fmt.Sprint(i))))
	}

	v := new(Int)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m := new(Map).Init()
		for _, k := range keys {
			m.Set(k, v)
		}
	}
}

func BenchmarkMapSetString(b *testing.B) {
	m := new(Map).Init()

	v := new(String)
	v.Set("Hello, !")

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m.Set("red", v)
		}
	})
}

func BenchmarkMapAddSame(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m := new(Map).Init()
			m.Add("red", 1)
			m.Add("red", 1)
			m.Add("red", 1)
			m.Add("red", 1)
		}
	})
}

func BenchmarkMapAddDifferent(b *testing.B) {
	procKeys := make([][]string, runtime.GOMAXPROCS(0))
	for i := range procKeys {
		keys := make([]string, 4)
		for j := range keys {
			keys[j] = fmt.Sprint(i, j)
		}
		procKeys[i] = keys
	}

	b.ResetTimer()

	var n int32
	b.RunParallel(func(pb *testing.PB) {
		i := int(atomic.AddInt32(&n, 1)-1) % len(procKeys)
		keys := procKeys[i]

		for pb.Next() {
			m := new(Map).Init()
			for _, k := range keys {
				m.Add(k, 1)
			}
		}
	})
}

// BenchmarkMapAddDifferentRandom simulates such a case where that the concerned
// keys of Map.Add are generated dynamically and as a result insertion is out of
// order and the number of the keys may be large.
func BenchmarkMapAddDifferentRandom(b *testing.B) {
	keys := make([]string, 100)
	for i := range keys {
		keys[i] = fmt.Sprintf("%x", sha1.Sum([]byte(fmt.Sprint(i))))
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m := new(Map).Init()
		for _, k := range keys {
			m.Add(k, 1)
		}
	}
}

func BenchmarkMapAddSameSteadyState(b *testing.B) {
	m := new(Map).Init()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m.Add("red", 1)
		}
	})
}

func BenchmarkMapAddDifferentSteadyState(b *testing.B) {
	procKeys := make([][]string, runtime.GOMAXPROCS(0))
	for i := range procKeys {
		keys := make([]string, 4)
		for j := range keys {
			keys[j] = fmt.Sprint(i, j)
		}
		procKeys[i] = keys
	}

	m := new(Map).Init()
	b.ResetTimer()

	var n int32
	b.RunParallel(func(pb *testing.PB) {
		i := int(atomic.AddInt32(&n, 1)-1) % len(procKeys)
		keys := procKeys[i]

		for pb.Next() {
			for _, k := range keys {
				m.Add(k, 1)
			}
		}
	})
}

func TestFunc(t *testing.T) {
	RemoveAll()
	var x any = []string{"a", "b"}
	f := Func(func() any { return x })
	if s, exp := f.String(), `["a","b"]`; s != exp {
		t.Errorf(`f.String() = %q, want %q`, s, exp)
	}
	if v := f.Value(); !reflect.DeepEqual(v, x) {
		t.Errorf(`f.Value() = %q, want %q`, v, x)
	}

	x = 17
	if s, exp := f.String(), `17`; s != exp {
		t.Errorf(`f.String() = %q, want %q`, s, exp)
	}
}

func TestHandler(t *testing.T) {
	RemoveAll()
	m := NewMap("map1")
	m.Add("a", 1)
	m.Add("z", 2)
	m2 := NewMap("map2")
	for i := 0; i < 9; i++ {
		m2.Add(strconv.Itoa(i), int64(i))
	}
	rr := httptest.NewRecorder()
	rr.Body = new(bytes.Buffer)
	expvarHandler(rr, nil)
	want := `{
"map1": {"a": 1, "z": 2},
"map2": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}
}
`
	if got := rr.Body.String(); got != want {
		t.Errorf("HTTP handler wrote:\n%s\nWant:\n%s", got, want)
	}
}

func BenchmarkMapString(b *testing.B) {
	var m, m1, m2 Map
	m.Set("map1", &m1)
	m1.Add("a", 1)
	m1.Add("z", 2)
	m.Set("map2", &m2)
	for i := 0; i < 9; i++ {
		m2.Add(strconv.Itoa(i), int64(i))
	}
	var s1, s2 String
	m.Set("str1", &s1)
	s1.Set("hello, world!")
	m.Set("str2", &s2)
	s2.Set("fizz buzz")
	b.ResetTimer()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.String()
	}
}

func BenchmarkRealworldExpvarUsage(b *testing.B) {
	var (
		bytesSent Int
		bytesRead Int
	)

	// The benchmark creates GOMAXPROCS client/server pairs.
	// Each pair creates 4 goroutines: client reader/writer and server reader/writer.
	// The benchmark stresses concurrent reading and writing to the same connection.
	// Such pattern is used in net/http and net/rpc.

	b.StopTimer()

	P := runtime.GOMAXPROCS(0)
	N := b.N / P
	W := 1000

	// Setup P client/server connections.
	clients := make([]net.Conn, P)
	servers := make([]net.Conn, P)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		b.Fatalf("Listen failed: %v", err)
	}
	defer ln.Close()
	done := make(chan bool, 1)
	go func() {
		for p := 0; p < P; p++ {
			s, err := ln.Accept()
			if err != nil {
				b.Errorf("Accept failed: %v", err)
				done <- false
				return
			}
			servers[p] = s
		}
		done <- true
	}()
	for p := 0; p < P; p++ {
		c, err := net.Dial("tcp", ln.Addr().String())
		if err != nil {
			<-done
			b.Fatalf("Dial failed: %v", err)
		}
		clients[p] = c
	}
	if !<-done {
		b.FailNow()
	}

	b.StartTimer()

	var wg sync.WaitGroup
	wg.Add(4 * P)
	for p := 0; p < P; p++ {
		// Client writer.
		go func(c net.Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				v := byte(i)
				for w := 0; w < W; w++ {
					v *= v
				}
				buf[0] = v
				n, err := c.Write(buf[:])
				if err != nil {
					b.Errorf("Write failed: %v", err)
					return
				}

				bytesSent.Add(int64(n))
			}
		}(clients[p])

		// Pipe between server reader and server writer.
		pipe := make(chan byte, 128)

		// Server reader.
		go func(s net.Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				n, err := s.Read(buf[:])

				if err != nil {
					b.Errorf("Read failed: %v", err)
					return
				}

				bytesRead.Add(int64(n))
				pipe <- buf[0]
			}
		}(servers[p])

		// Server writer.
		go func(s net.Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				v := <-pipe
				for w := 0; w < W; w++ {
					v *= v
				}
				buf[0] = v
				n, err := s.Write(buf[:])
				if err != nil {
					b.Errorf("Write failed: %v", err)
					return
				}

				bytesSent.Add(int64(n))
			}
			s.Close()
		}(servers[p])

		// Client reader.
		go func(c net.Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				n, err := c.Read(buf[:])

				if err != nil {
					b.Errorf("Read failed: %v", err)
					return
				}

				bytesRead.Add(int64(n))
			}
			c.Close()
		}(clients[p])
	}
	wg.Wait()
}

func TestAppendJSONQuote(t *testing.T) {
	var b []byte
	for i := 0; i < 128; i++ {
		b = append(b, byte(i))
	}
	b = append(b, "\u2028\u2029"...)
	got := string(appendJSONQuote(nil, string(b[:])))
	want := `"` +
		`\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008\t\n\u000b\u000c\r\u000e\u000f` +
		`\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f` +
		` !\"#$%\u0026'()*+,-./0123456789:;\u003c=\u003e?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_` +
		"`" + `abcdefghijklmnopqrstuvwxyz{|}~` + "\x7f" + `\u2028\u2029"`
	if got != want {
		t.Errorf("appendJSONQuote mismatch:\ngot  %v\nwant %v", got, want)
	}
}
