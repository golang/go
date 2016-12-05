// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"unsafe"
)

type RWS struct{}

func (x *RWS) Read(p []byte) (n int, err error)                   { return }
func (x *RWS) Write(p []byte) (n int, err error)                  { return }
func (x *RWS) Seek(offset int64, whence int) (n int64, err error) { return }
func (x *RWS) String() string                                     { return "rws" }

func makeRWS() io.ReadWriteSeeker { return &RWS{} }
func makeStringer() fmt.Stringer  { return &RWS{} }

// Test correct construction of static empty interface values
var efaces = [...]struct {
	x interface{}
	s string
}{
	{nil, "<nil> <nil>"},
	{1, "int 1"},
	{int(1), "int 1"},
	{Int(int(2)), "main.Int Int=2"},
	{int(Int(3)), "int 3"},
	{[1]int{2}, "[1]int [2]"},
	{io.Reader(io.ReadWriter(io.ReadWriteSeeker(nil))), "<nil> <nil>"},
	{io.Reader(io.ReadWriter(io.ReadWriteSeeker(&RWS{}))), "*main.RWS rws"},
	{makeRWS(), "*main.RWS rws"},
	{map[string]string{"here": "there"}, "map[string]string map[here:there]"},
	{chan bool(nil), "chan bool <nil>"},
	{unsafe.Pointer(uintptr(0)), "unsafe.Pointer <nil>"},
	{(*byte)(nil), "*uint8 <nil>"},
	{io.Writer((*os.File)(nil)), "*os.File <nil>"},
	{(interface{})(io.Writer((*os.File)(nil))), "*os.File <nil>"},
	{fmt.Stringer(Strunger(((*Int)(nil)))), "*main.Int <nil>"},
}

type Int int

func (i Int) String() string { return fmt.Sprintf("Int=%d", i) }
func (i Int) Strung()        {}

type Strunger interface {
	fmt.Stringer
	Strung()
}

// Test correct construction of static non-empty interface values
var ifaces = [...]struct {
	x fmt.Stringer
	s string
}{
	{nil, "<nil> <nil> %!s(<nil>)"},
	{Int(3), "main.Int 3 Int=3"},
	{Int(int(Int(4))), "main.Int 4 Int=4"},
	{Strunger(Int(5)), "main.Int 5 Int=5"},
	{makeStringer(), "*main.RWS &main.RWS{} rws"},
	{fmt.Stringer(nil), "<nil> <nil> %!s(<nil>)"},
	{(*RWS)(nil), "*main.RWS (*main.RWS)(nil) rws"},
}

// Test correct handling of direct interface values
var (
	one  int         = 1
	iptr interface{} = &one
	clos int
	f    interface{} = func() { clos++ }
	deep interface{} = [1]struct{ a *[2]byte }{{a: &[2]byte{'z', 'w'}}}
	ch   interface{} = make(chan bool, 1)
)

func main() {
	var fail bool
	for i, test := range efaces {
		s := fmt.Sprintf("%[1]T %[1]v", test.x)
		if s != test.s {
			fmt.Printf("eface(%d)=%q want %q\n", i, s, test.s)
			fail = true
		}
	}

	for i, test := range ifaces {
		s := fmt.Sprintf("%[1]T %#[1]v %[1]s", test.x)
		if s != test.s {
			fmt.Printf("iface(%d)=%q want %q\n", i, s, test.s)
			fail = true
		}
	}

	if got := *(iptr.(*int)); got != 1 {
		fmt.Printf("bad int ptr %d\n", got)
		fail = true
	}

	f.(func())()
	f.(func())()
	f.(func())()
	if clos != 3 {
		fmt.Printf("bad closure exec %d\n", clos)
		fail = true
	}

	if !reflect.DeepEqual(*(deep.([1]struct{ a *[2]byte })[0].a), [2]byte{'z', 'w'}) {
		fmt.Printf("bad deep directiface\n")
		fail = true
	}

	cc := ch.(chan bool)
	cc <- true
	if got := <-cc; !got {
		fmt.Printf("bad chan\n")
		fail = true
	}

	if fail {
		fmt.Println("BUG")
		os.Exit(1)
	}
}
