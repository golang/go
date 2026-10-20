// errorcheck

package p

import (
	"bufio"
	"io"
	"os"
)

type wrapper struct{ f *os.File }

type box struct{ n int }

var registry = map[string]*os.File{}

func newBox() *box { return &box{} }

func consume(*os.File) {}

func mustOpen() *os.File {
	f, err := os.Open("x")
	if err != nil {
		panic(err)
	}
	return f
}

func plainUse() {
	f := mustOpen() // ERROR "unclosed_io_closer"
	_ = f.Name()
}

func pairUse() error {
	f, err := os.Open("x") // ERROR "unclosed_io_closer"
	if err != nil {
		return err
	}
	_ = f.Name()
	return nil
}

func explicitClose() {
	f := mustOpen() // ERROR "unclosed_io_closer"
	if err := f.Close(); err != nil {
		panic(err)
	}
}

func declType() {
	var c io.Closer = mustOpen() // ERROR "unclosed_io_closer"
	if c == nil {
		panic("nil")
	}
}

func laterAssign() error {
	var c io.Closer
	var err error
	c, err = os.Open("x") // ERROR "unclosed_io_closer"
	if err != nil {
		return err
	}
	if c == nil {
		panic("nil")
	}
	return nil
}

func inLiteral() {
	func() {
		f := mustOpen() // ERROR "unclosed_io_closer"
		_ = f.Name()
	}()
}

func nopCloser(r io.Reader) {
	c := io.NopCloser(r) // ERROR "unclosed_io_closer"
	if c == nil {
		panic("nil")
	}
}

func reassigned() {
	f := mustOpen() // ERROR "unclosed_io_closer"
	f = mustOpen()
	_ = f.Name()
}

func wrongVar() {
	g := mustOpen()
	f := mustOpen() // ERROR "unclosed_io_closer"
	defer g.Close()
	_ = f.Name()
}

func compareOnly() {
	f := mustOpen() // ERROR "unclosed_io_closer"
	if f != nil {
		_ = f.Name()
	}
}

func blankDiscards() {
	f := mustOpen() // ERROR "unclosed_io_closer"
	_ = f
}

func blankDiscardsPair() {
	f, err := os.Open("x") // ERROR "unclosed_io_closer"
	if err != nil {
		panic(err)
	}
	_, _ = err, f
}

func shadow() {
	f := mustOpen()
	defer f.Close()
	{
		f := mustOpen() // ERROR "unclosed_io_closer"
		_ = f.Name()
	}
}

func deferred() {
	f := mustOpen()
	defer f.Close()
	_ = f.Name()
}

func deferredInIf(ok bool) {
	f := mustOpen()
	if ok {
		defer f.Close()
	}
	_ = f.Name()
}

func deferredLiteral() {
	f := mustOpen()
	defer func() {
		if err := f.Close(); err != nil {
			panic(err)
		}
	}()
	_ = f.Name()
}

func returned() *os.File {
	f := mustOpen()
	return f
}

func returnedPair() (*os.File, error) {
	f, err := os.Open("x")
	return f, err
}

func wrapped() *wrapper {
	f := mustOpen()
	return &wrapper{f}
}

func namedResult() (f *os.File, err error) {
	f, err = os.Open("x")
	return
}

func storedField(w *wrapper) {
	f := mustOpen()
	w.f = f
}

func storedMap() {
	f := mustOpen()
	registry["k"] = f
}

func sent() {
	f := mustOpen()
	ch := make(chan *os.File, 1)
	ch <- f
}

func appended() {
	var list []*os.File
	f := mustOpen()
	list = append(list, f)
	_ = list
}

func passedOn() {
	f := mustOpen()
	consume(f)
}

func buffered() *bufio.Reader {
	f := mustOpen()
	return bufio.NewReader(f)
}

func captured() func() error {
	f := mustOpen()
	return func() error { return f.Close() }
}

func methodValue() func() error {
	f := mustOpen()
	cleanup := f.Close
	return cleanup
}

func aliased() *os.File {
	f := mustOpen()
	g := f
	return g
}

func notCloser() {
	b := newBox()
	_ = b.n
}

func notFromCall() {
	var f io.Closer
	g := os.Stdout
	h := &os.File{}
	_, _, _ = f, g, h
}

func param(f *os.File) {
	_ = f.Name()
}

func conversion(f *os.File) {
	c := io.Closer(f)
	_ = c
}

//logos:allow(correctness::unclosed_io_closer)
func allowedClose() {
	f := mustOpen()
	_ = f.Name()
}
