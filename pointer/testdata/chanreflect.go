// +build ignore

package main

import "reflect"

// Test of channels with reflection.

var a, b int

func chanreflect1() {
	ch := make(chan *int, 0) // @line cr1make
	crv := reflect.ValueOf(ch)
	crv.Send(reflect.ValueOf(&a))
	print(crv.Interface())             // @types chan *int
	print(crv.Interface().(chan *int)) // @pointsto makechan@cr1make:12
	print(<-ch)                        // @pointsto main.a
}

func chanreflect2() {
	ch := make(chan *int, 0)
	ch <- &b
	crv := reflect.ValueOf(ch)
	r, _ := crv.Recv()
	print(r.Interface())        // @types *int
	print(r.Interface().(*int)) // @pointsto main.b
}

// TODO(adonovan): the analysis can't yet take advantage of the
// ChanOf(dir) parameter so the results are less precise than they
// should be: all three directions are returned.

func chanOfRecv() {
	// MakeChan(<-chan) is a no-op.
	t := reflect.ChanOf(reflect.RecvDir, reflect.TypeOf(&a))
	print(reflect.Zero(t).Interface())                      // @types <-chan *int | chan<- *int | chan *int
	print(reflect.MakeChan(t, 0).Interface().(<-chan *int)) // @pointsto
	print(reflect.MakeChan(t, 0).Interface().(chan *int))   // @pointsto <alloc in reflect.MakeChan>
}

func chanOfSend() {
	// MakeChan(chan<-) is a no-op.
	t := reflect.ChanOf(reflect.SendDir, reflect.TypeOf(&a))
	print(reflect.Zero(t).Interface())                      // @types <-chan *int | chan<- *int | chan *int
	print(reflect.MakeChan(t, 0).Interface().(chan<- *int)) // @pointsto
	print(reflect.MakeChan(t, 0).Interface().(chan *int))   // @pointsto <alloc in reflect.MakeChan>
}

func chanOfBoth() {
	t := reflect.ChanOf(reflect.BothDir, reflect.TypeOf(&a))
	print(reflect.Zero(t).Interface()) // @types <-chan *int | chan<- *int | chan *int
	ch := reflect.MakeChan(t, 0)
	print(ch.Interface().(chan *int)) // @pointsto <alloc in reflect.MakeChan>
	ch.Send(reflect.ValueOf(&b))
	ch.Interface().(chan *int) <- &a
	r, _ := ch.Recv()
	print(r.Interface().(*int))         // @pointsto main.a | main.b
	print(<-ch.Interface().(chan *int)) // @pointsto main.a | main.b
}

var unknownDir reflect.ChanDir // not a constant

func chanOfUnknown() {
	// Unknown channel direction: assume all three.
	// MakeChan only works on the bi-di channel type.
	t := reflect.ChanOf(unknownDir, reflect.TypeOf(&a))
	print(reflect.Zero(t).Interface())        // @types <-chan *int | chan<- *int | chan *int
	print(reflect.MakeChan(t, 0).Interface()) // @types chan *int
}

func main() {
	chanreflect1()
	chanreflect2()
	chanOfRecv()
	chanOfSend()
	chanOfBoth()
	chanOfUnknown()
}
