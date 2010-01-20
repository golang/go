// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netchan

import (
	"fmt"
	"testing"
)

type value struct {
	i int
	s string
}

func exportSend(exp *Exporter, t *testing.T) {
	c := make(chan value)
	err := exp.Export("name", c, Send)
	if err != nil {
		t.Fatal("export:", err)
	}
	c <- value{23, "hello"}
}

func importReceive(imp *Importer, t *testing.T) {
	ch := make(chan value)
	err := imp.ImportNValues("name", ch, Recv, new(value), 1)
	if err != nil {
		t.Fatal("import:", err)
	}
	v := <-ch
	fmt.Printf("%v\n", v)
	if v.i != 23 || v.s != "hello" {
		t.Errorf("bad value: expected 23, hello; got %+v\n", v)
	}
}

func TestBabyStep(t *testing.T) {
	exp, err := NewExporter("tcp", ":0")
	if err != nil {
		t.Fatal("new exporter:", err)
	}
	go exportSend(exp, t)
	imp, err := NewImporter("tcp", exp.Addr().String())
	if err != nil {
		t.Fatal("new importer:", err)
	}
	importReceive(imp, t)
}
