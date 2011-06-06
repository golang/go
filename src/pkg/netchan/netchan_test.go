// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"net"
	"strings"
	"testing"
	"time"
)

const count = 10     // number of items in most tests
const closeCount = 5 // number of items when sender closes early

const base = 23

func exportSend(exp *Exporter, n int, t *testing.T, done chan bool) {
	ch := make(chan int)
	err := exp.Export("exportedSend", ch, Send)
	if err != nil {
		t.Fatal("exportSend:", err)
	}
	go func() {
		for i := 0; i < n; i++ {
			ch <- base + i
		}
		close(ch)
		if done != nil {
			done <- true
		}
	}()
}

func exportReceive(exp *Exporter, t *testing.T, expDone chan bool) {
	ch := make(chan int)
	err := exp.Export("exportedRecv", ch, Recv)
	expDone <- true
	if err != nil {
		t.Fatal("exportReceive:", err)
	}
	for i := 0; i < count; i++ {
		v, ok := <-ch
		if !ok {
			if i != closeCount {
				t.Errorf("exportReceive expected close at %d; got one at %d", closeCount, i)
			}
			break
		}
		if v != base+i {
			t.Errorf("export Receive: bad value: expected %d+%d=%d; got %d", base, i, base+i, v)
		}
	}
}

func importSend(imp *Importer, n int, t *testing.T, done chan bool) {
	ch := make(chan int)
	err := imp.ImportNValues("exportedRecv", ch, Send, 3, -1)
	if err != nil {
		t.Fatal("importSend:", err)
	}
	go func() {
		for i := 0; i < n; i++ {
			ch <- base + i
		}
		close(ch)
		if done != nil {
			done <- true
		}
	}()
}

func importReceive(imp *Importer, t *testing.T, done chan bool) {
	ch := make(chan int)
	err := imp.ImportNValues("exportedSend", ch, Recv, 3, count)
	if err != nil {
		t.Fatal("importReceive:", err)
	}
	for i := 0; i < count; i++ {
		v, ok := <-ch
		if !ok {
			if i != closeCount {
				t.Errorf("importReceive expected close at %d; got one at %d", closeCount, i)
			}
			break
		}
		if v != base+i {
			t.Errorf("importReceive: bad value: expected %d+%d=%d; got %+d", base, i, base+i, v)
		}
	}
	if done != nil {
		done <- true
	}
}

func TestExportSendImportReceive(t *testing.T) {
	exp, imp := pair(t)
	exportSend(exp, count, t, nil)
	importReceive(imp, t, nil)
}

func TestExportReceiveImportSend(t *testing.T) {
	exp, imp := pair(t)
	expDone := make(chan bool)
	done := make(chan bool)
	go func() {
		exportReceive(exp, t, expDone)
		done <- true
	}()
	<-expDone
	importSend(imp, count, t, nil)
	<-done
}

func TestClosingExportSendImportReceive(t *testing.T) {
	exp, imp := pair(t)
	exportSend(exp, closeCount, t, nil)
	importReceive(imp, t, nil)
}

func TestClosingImportSendExportReceive(t *testing.T) {
	exp, imp := pair(t)
	expDone := make(chan bool)
	done := make(chan bool)
	go func() {
		exportReceive(exp, t, expDone)
		done <- true
	}()
	<-expDone
	importSend(imp, closeCount, t, nil)
	<-done
}

func TestErrorForIllegalChannel(t *testing.T) {
	exp, imp := pair(t)
	// Now export a channel.
	ch := make(chan int, 1)
	err := exp.Export("aChannel", ch, Send)
	if err != nil {
		t.Fatal("export:", err)
	}
	ch <- 1234
	close(ch)
	// Now try to import a different channel.
	ch = make(chan int)
	err = imp.Import("notAChannel", ch, Recv, 1)
	if err != nil {
		t.Fatal("import:", err)
	}
	// Expect an error now.  Start a timeout.
	timeout := make(chan bool, 1) // buffered so closure will not hang around.
	go func() {
		time.Sleep(10e9) // very long, to give even really slow machines a chance.
		timeout <- true
	}()
	select {
	case err = <-imp.Errors():
		if strings.Index(err.String(), "no such channel") < 0 {
			t.Error("wrong error for nonexistent channel:", err)
		}
	case <-timeout:
		t.Error("import of nonexistent channel did not receive an error")
	}
}

// Not a great test but it does at least invoke Drain.
func TestExportDrain(t *testing.T) {
	exp, imp := pair(t)
	done := make(chan bool)
	go func() {
		exportSend(exp, closeCount, t, nil)
		done <- true
	}()
	<-done
	go importReceive(imp, t, done)
	exp.Drain(0)
	<-done
}

// Not a great test but it does at least invoke Drain.
func TestImportDrain(t *testing.T) {
	exp, imp := pair(t)
	expDone := make(chan bool)
	go exportReceive(exp, t, expDone)
	<-expDone
	importSend(imp, closeCount, t, nil)
	imp.Drain(0)
}

// Not a great test but it does at least invoke Sync.
func TestExportSync(t *testing.T) {
	exp, imp := pair(t)
	done := make(chan bool)
	exportSend(exp, closeCount, t, nil)
	go importReceive(imp, t, done)
	exp.Sync(0)
	<-done
}

// Test hanging up the send side of an export.
// TODO: test hanging up the receive side of an export.
func TestExportHangup(t *testing.T) {
	exp, imp := pair(t)
	ech := make(chan int)
	err := exp.Export("exportedSend", ech, Send)
	if err != nil {
		t.Fatal("export:", err)
	}
	// Prepare to receive two values. We'll actually deliver only one.
	ich := make(chan int)
	err = imp.ImportNValues("exportedSend", ich, Recv, 1, 2)
	if err != nil {
		t.Fatal("import exportedSend:", err)
	}
	// Send one value, receive it.
	const Value = 1234
	ech <- Value
	v := <-ich
	if v != Value {
		t.Fatal("expected", Value, "got", v)
	}
	// Now hang up the channel.  Importer should see it close.
	exp.Hangup("exportedSend")
	v, ok := <-ich
	if ok {
		t.Fatal("expected channel to be closed; got value", v)
	}
}

// Test hanging up the send side of an import.
// TODO: test hanging up the receive side of an import.
func TestImportHangup(t *testing.T) {
	exp, imp := pair(t)
	ech := make(chan int)
	err := exp.Export("exportedRecv", ech, Recv)
	if err != nil {
		t.Fatal("export:", err)
	}
	// Prepare to Send two values. We'll actually deliver only one.
	ich := make(chan int)
	err = imp.ImportNValues("exportedRecv", ich, Send, 1, 2)
	if err != nil {
		t.Fatal("import exportedRecv:", err)
	}
	// Send one value, receive it.
	const Value = 1234
	ich <- Value
	v := <-ech
	if v != Value {
		t.Fatal("expected", Value, "got", v)
	}
	// Now hang up the channel.  Exporter should see it close.
	imp.Hangup("exportedRecv")
	v, ok := <-ech
	if ok {
		t.Fatal("expected channel to be closed; got value", v)
	}
}

// loop back exportedRecv to exportedSend,
// but receive a value from ctlch before starting the loop.
func exportLoopback(exp *Exporter, t *testing.T) {
	inch := make(chan int)
	if err := exp.Export("exportedRecv", inch, Recv); err != nil {
		t.Fatal("exportRecv")
	}

	outch := make(chan int)
	if err := exp.Export("exportedSend", outch, Send); err != nil {
		t.Fatal("exportSend")
	}

	ctlch := make(chan int)
	if err := exp.Export("exportedCtl", ctlch, Recv); err != nil {
		t.Fatal("exportRecv")
	}

	go func() {
		<-ctlch
		for i := 0; i < count; i++ {
			x := <-inch
			if x != base+i {
				t.Errorf("exportLoopback expected %d; got %d", i, x)
			}
			outch <- x
		}
	}()
}

// This test checks that channel operations can proceed
// even when other concurrent operations are blocked.
func TestIndependentSends(t *testing.T) {
	exp, imp := pair(t)

	exportLoopback(exp, t)

	importSend(imp, count, t, nil)
	done := make(chan bool)
	go importReceive(imp, t, done)

	// wait for export side to try to deliver some values.
	time.Sleep(0.25e9)

	ctlch := make(chan int)
	if err := imp.ImportNValues("exportedCtl", ctlch, Send, 1, 1); err != nil {
		t.Fatal("importSend:", err)
	}
	ctlch <- 0

	<-done
}

// This test cross-connects a pair of exporter/importer pairs.
type value struct {
	I      int
	Source string
}

func TestCrossConnect(t *testing.T) {
	e1, i1 := pair(t)
	e2, i2 := pair(t)

	crossExport(e1, e2, t)
	crossImport(i1, i2, t)
}

// Export side of cross-traffic.
func crossExport(e1, e2 *Exporter, t *testing.T) {
	s := make(chan value)
	err := e1.Export("exportedSend", s, Send)
	if err != nil {
		t.Fatal("exportSend:", err)
	}

	r := make(chan value)
	err = e2.Export("exportedReceive", r, Recv)
	if err != nil {
		t.Fatal("exportReceive:", err)
	}

	go crossLoop("export", s, r, t)
}

// Import side of cross-traffic.
func crossImport(i1, i2 *Importer, t *testing.T) {
	s := make(chan value)
	err := i2.Import("exportedReceive", s, Send, 2)
	if err != nil {
		t.Fatal("import of exportedReceive:", err)
	}

	r := make(chan value)
	err = i1.Import("exportedSend", r, Recv, 2)
	if err != nil {
		t.Fatal("import of exported Send:", err)
	}

	crossLoop("import", s, r, t)
}

// Cross-traffic: send and receive 'count' numbers.
func crossLoop(name string, s, r chan value, t *testing.T) {
	for si, ri := 0, 0; si < count && ri < count; {
		select {
		case s <- value{si, name}:
			si++
		case v := <-r:
			if v.I != ri {
				t.Errorf("loop: bad value: expected %d, hello; got %+v", ri, v)
			}
			ri++
		}
	}
}

const flowCount = 100

// test flow control from exporter to importer.
func TestExportFlowControl(t *testing.T) {
	exp, imp := pair(t)

	sendDone := make(chan bool, 1)
	exportSend(exp, flowCount, t, sendDone)

	ch := make(chan int)
	err := imp.ImportNValues("exportedSend", ch, Recv, 20, -1)
	if err != nil {
		t.Fatal("importReceive:", err)
	}

	testFlow(sendDone, ch, flowCount, t)
}

// test flow control from importer to exporter.
func TestImportFlowControl(t *testing.T) {
	exp, imp := pair(t)

	ch := make(chan int)
	err := exp.Export("exportedRecv", ch, Recv)
	if err != nil {
		t.Fatal("importReceive:", err)
	}

	sendDone := make(chan bool, 1)
	importSend(imp, flowCount, t, sendDone)
	testFlow(sendDone, ch, flowCount, t)
}

func testFlow(sendDone chan bool, ch <-chan int, N int, t *testing.T) {
	go func() {
		time.Sleep(0.5e9)
		sendDone <- false
	}()

	if <-sendDone {
		t.Fatal("send did not block")
	}
	n := 0
	for i := range ch {
		t.Log("after blocking, got value ", i)
		n++
	}
	if n != N {
		t.Fatalf("expected %d values; got %d", N, n)
	}
}

func pair(t *testing.T) (*Exporter, *Importer) {
	c0, c1 := net.Pipe()
	exp := NewExporter()
	go exp.ServeConn(c0)
	imp := NewImporter(c1)
	return exp, imp
}
