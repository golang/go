// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"strings"
	"testing"
	"time"
)

const count = 10     // number of items in most tests
const closeCount = 5 // number of items when sender closes early

const base = 23

func exportSend(exp *Exporter, n int, t *testing.T) {
	ch := make(chan int)
	err := exp.Export("exportedSend", ch, Send)
	if err != nil {
		t.Fatal("exportSend:", err)
	}
	go func() {
		for i := 0; i < n; i++ {
			ch <- base+i
		}
		close(ch)
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
		v := <-ch
		if closed(ch) {
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

func importSend(imp *Importer, n int, t *testing.T) {
	ch := make(chan int)
	err := imp.ImportNValues("exportedRecv", ch, Send, count)
	if err != nil {
		t.Fatal("importSend:", err)
	}
	go func() {
		for i := 0; i < n; i++ {
			ch <- base+i
		}
		close(ch)
	}()
}

func importReceive(imp *Importer, t *testing.T, done chan bool) {
	ch := make(chan int)
	err := imp.ImportNValues("exportedSend", ch, Recv, count)
	if err != nil {
		t.Fatal("importReceive:", err)
	}
	for i := 0; i < count; i++ {
		v := <-ch
		if closed(ch) {
			if i != closeCount {
				t.Errorf("importReceive expected close at %d; got one at %d", closeCount, i)
			}
			break
		}
		if v != 23+i {
			t.Errorf("importReceive: bad value: expected %%d+%d=%d; got %+d", base, i, base+i, v)
		}
	}
	if done != nil {
		done <- true
	}
}

func TestExportSendImportReceive(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	exportSend(exp, count, t)
	importReceive(imp, t, nil)
}

func TestExportReceiveImportSend(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	expDone := make(chan bool)
	done := make(chan bool)
	go func() {
		exportReceive(exp, t, expDone)
		done <- true
	}()
	<-expDone
	importSend(imp, count, t)
	<-done
}

func TestClosingExportSendImportReceive(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	exportSend(exp, closeCount, t)
	importReceive(imp, t, nil)
}

func TestClosingImportSendExportReceive(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	expDone := make(chan bool)
	done := make(chan bool)
	go func() {
		exportReceive(exp, t, expDone)
		done <- true
	}()
	<-expDone
	importSend(imp, closeCount, t)
	<-done
}

func TestErrorForIllegalChannel(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	// Now export a channel.
	ch := make(chan int, 1)
	err = exp.Export("aChannel", ch, Send)
	if err != nil {
		t.Fatal("export:", err)
	}
	ch <- 1234
	close(ch)
	// Now try to import a different channel.
	ch = make(chan int)
	err = imp.Import("notAChannel", ch, Recv)
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
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	done := make(chan bool)
	go func() {
		exportSend(exp, closeCount, t)
		done <- true
	}()
	<-done
	go importReceive(imp, t, done)
	exp.Drain(0)
	<-done
}

// Not a great test but it does at least invoke Sync.
func TestExportSync(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	done := make(chan bool)
	exportSend(exp, closeCount, t)
	go importReceive(imp, t, done)
	exp.Sync(0)
	<-done
}

// Test hanging up the send side of an export.
// TODO: test hanging up the receive side of an export.
func TestExportHangup(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	ech := make(chan int)
	err = exp.Export("exportedSend", ech, Send)
	if err != nil {
		t.Fatal("export:", err)
	}
	// Prepare to receive two values. We'll actually deliver only one.
	ich := make(chan int)
	err = imp.ImportNValues("exportedSend", ich, Recv, 2)
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
	v = <-ich
	if !closed(ich) {
		t.Fatal("expected channel to be closed; got value", v)
	}
}

// Test hanging up the send side of an import.
// TODO: test hanging up the receive side of an import.
func TestImportHangup(t *testing.T) {
	exp, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	ech := make(chan int)
	err = exp.Export("exportedRecv", ech, Recv)
	if err != nil {
		t.Fatal("export:", err)
	}
	// Prepare to Send two values. We'll actually deliver only one.
	ich := make(chan int)
	err = imp.ImportNValues("exportedRecv", ich, Send, 2)
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
	v = <-ech
	if !closed(ech) {
		t.Fatal("expected channel to be closed; got value", v)
	}
}

// This test cross-connects a pair of exporter/importer pairs.
type value struct {
	i      int
	source string
}

func TestCrossConnect(t *testing.T) {
	e1, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	i1, err := NewImporter("tcp", e1.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}

	e2, err := NewExporter("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	i2, err := NewImporter("tcp", e2.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}

	go crossExport(e1, e2, t)
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

	crossLoop("export", s, r, t)
}

// Import side of cross-traffic.
func crossImport(i1, i2 *Importer, t *testing.T) {
	s := make(chan value)
	err := i2.Import("exportedReceive", s, Send)
	if err != nil {
		t.Fatal("import of exportedReceive:", err)
	}

	r := make(chan value)
	err = i1.Import("exportedSend", r, Recv)
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
			if v.i != ri {
				t.Errorf("loop: bad value: expected %d, hello; got %+v", ri, v)
			}
			ri++
		}
	}
}
