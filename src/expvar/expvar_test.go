// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package expvar

import (
	"bytes"
	"encoding/json"
	"math"
	"net"
	"net/http/httptest"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
)

// RemoveAll removes all exported variables.
// This is for tests only.
func RemoveAll() {
	mutex.Lock()
	defer mutex.Unlock()
	vars = make(map[string]Var)
	varKeys = nil
}

func TestInt(t *testing.T) {
	RemoveAll()
	reqs := NewInt("requests")
	if reqs.i != 0 {
		t.Errorf("reqs.i = %v, want 0", reqs.i)
	}
	if reqs != Get("requests").(*Int) {
		t.Errorf("Get() failed.")
	}

	reqs.Add(1)
	reqs.Add(3)
	if reqs.i != 4 {
		t.Errorf("reqs.i = %v, want 4", reqs.i)
	}

	if s := reqs.String(); s != "4" {
		t.Errorf("reqs.String() = %q, want \"4\"", s)
	}

	reqs.Set(-2)
	if reqs.i != -2 {
		t.Errorf("reqs.i = %v, want -2", reqs.i)
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

func (v *Float) val() float64 {
	return math.Float64frombits(atomic.LoadUint64(&v.f))
}

func TestFloat(t *testing.T) {
	RemoveAll()
	reqs := NewFloat("requests-float")
	if reqs.f != 0.0 {
		t.Errorf("reqs.f = %v, want 0", reqs.f)
	}
	if reqs != Get("requests-float").(*Float) {
		t.Errorf("Get() failed.")
	}

	reqs.Add(1.5)
	reqs.Add(1.25)
	if v := reqs.val(); v != 2.75 {
		t.Errorf("reqs.val() = %v, want 2.75", v)
	}

	if s := reqs.String(); s != "2.75" {
		t.Errorf("reqs.String() = %q, want \"4.64\"", s)
	}

	reqs.Add(-2)
	if v := reqs.val(); v != 0.75 {
		t.Errorf("reqs.val() = %v, want 0.75", v)
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
	if name.s != "" {
		t.Errorf("name.s = %q, want \"\"", name.s)
	}

	name.Set("Mike")
	if name.s != "Mike" {
		t.Errorf("name.s = %q, want \"Mike\"", name.s)
	}

	if s := name.String(); s != "\"Mike\"" {
		t.Errorf("reqs.String() = %q, want \"\"Mike\"\"", s)
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

func TestMapCounter(t *testing.T) {
	RemoveAll()
	colors := NewMap("bike-shed-colors")

	colors.Add("red", 1)
	colors.Add("red", 2)
	colors.Add("blue", 4)
	colors.AddFloat(`green "midori"`, 4.125)
	if x := colors.m["red"].(*Int).i; x != 3 {
		t.Errorf("colors.m[\"red\"] = %v, want 3", x)
	}
	if x := colors.m["blue"].(*Int).i; x != 4 {
		t.Errorf("colors.m[\"blue\"] = %v, want 4", x)
	}
	if x := colors.m[`green "midori"`].(*Float).val(); x != 4.125 {
		t.Errorf("colors.m[`green \"midori\"] = %v, want 4.125", x)
	}

	// colors.String() should be '{"red":3, "blue":4}',
	// though the order of red and blue could vary.
	s := colors.String()
	var j interface{}
	err := json.Unmarshal([]byte(s), &j)
	if err != nil {
		t.Errorf("colors.String() isn't valid JSON: %v", err)
	}
	m, ok := j.(map[string]interface{})
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

func BenchmarkMapSet(b *testing.B) {
	m := new(Map).Init()

	v := new(Int)

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m.Set("red", v)
		}
	})
}

func BenchmarkMapAddSame(b *testing.B) {
	for i := 0; i < b.N; i++ {
		m := new(Map).Init()
		m.Add("red", 1)
		m.Add("red", 1)
		m.Add("red", 1)
		m.Add("red", 1)
	}
}

func BenchmarkMapAddDifferent(b *testing.B) {
	for i := 0; i < b.N; i++ {
		m := new(Map).Init()
		m.Add("red", 1)
		m.Add("blue", 1)
		m.Add("green", 1)
		m.Add("yellow", 1)
	}
}

func TestFunc(t *testing.T) {
	RemoveAll()
	var x interface{} = []string{"a", "b"}
	f := Func(func() interface{} { return x })
	if s, exp := f.String(), `["a","b"]`; s != exp {
		t.Errorf(`f.String() = %q, want %q`, s, exp)
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
	done := make(chan bool)
	go func() {
		for p := 0; p < P; p++ {
			s, err := ln.Accept()
			if err != nil {
				b.Errorf("Accept failed: %v", err)
				return
			}
			servers[p] = s
		}
		done <- true
	}()
	for p := 0; p < P; p++ {
		c, err := net.Dial("tcp", ln.Addr().String())
		if err != nil {
			b.Fatalf("Dial failed: %v", err)
		}
		clients[p] = c
	}
	<-done

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
