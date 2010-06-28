// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import "testing"

const count = 10     // number of items in most tests
const closeCount = 5 // number of items when sender closes early

func exportSend(exp *Exporter, n int, t *testing.T) {
	ch := make(chan int)
	err := exp.Export("exportedSend", ch, Send, new(int))
	if err != nil {
		t.Fatal("exportSend:", err)
	}
	for i := 0; i < n; i++ {
		ch <- 23+i
	}
	close(ch)
}

func exportReceive(exp *Exporter, t *testing.T) {
	ch := make(chan int)
	err := exp.Export("exportedRecv", ch, Recv, new(int))
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

func importReceive(imp *Importer, t *testing.T) {
	ch := make(chan int)
	err := imp.ImportNValues("exportedSend", ch, Recv, new(int), count)
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
}

func importSend(imp *Importer, t *testing.T) {
	ch := make(chan int)
	err := imp.ImportNValues("exportedRecv", ch, Send, new(int), count)
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
	go exportSend(exp, count, t)
	importReceive(imp, t)
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
	go exportSend(exp, closeCount, t)
	importReceive(imp, t)
}
