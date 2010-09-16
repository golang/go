// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import "testing"

const count = 10     // number of items in most tests
const closeCount = 5 // number of items when sender closes early

func exportSend(exp *Exporter, n int, t *testing.T) {
	ch := make(chan int)
	err := exp.Export("exportedSend", ch, Send)
	if err != nil {
		t.Fatal("exportSend:", err)
	}
	go func() {
		for i := 0; i < n; i++ {
			ch <- 23+i
		}
		close(ch)
	}()
}

func exportReceive(exp *Exporter, t *testing.T) {
	ch := make(chan int)
	err := exp.Export("exportedRecv", ch, Recv)
	if err != nil {
		t.Fatal("exportReceive:", err)
	}
	for i := 0; i < count; i++ {
		v := <-ch
		if v != 45+i {
			t.Errorf("export Receive: bad value: expected 4%d; got %d", 45+i, v)
		}
	}
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
				t.Errorf("expected close at %d; got one at %d\n", closeCount, i)
			}
			break
		}
		if v != 23+i {
			t.Errorf("importReceive: bad value: expected %d; got %+d", 23+i, v)
		}
	}
	if done != nil {
		done <- true
	}
}

func importSend(imp *Importer, t *testing.T) {
	ch := make(chan int)
	err := imp.ImportNValues("exportedRecv", ch, Send, count)
	if err != nil {
		t.Fatal("importSend:", err)
	}
	for i := 0; i < count; i++ {
		ch <- 45+i
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
	go importSend(imp, t)
	exportReceive(exp, t)
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
	go exportSend(exp, closeCount, t)
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
	go importReceive(imp, t, done)
	exportSend(exp, closeCount, t)
	exp.Sync(0)
	<-done
}

type value struct {
	i      int
	source string
}

// This test cross-connects a pair of exporter/importer pairs.
func TestCrossConnect(t *testing.T) {
	e1, err := NewExporter("tcp", ":0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	i1, err := NewImporter("tcp", e1.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}

	e2, err := NewExporter("tcp", ":0")
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
