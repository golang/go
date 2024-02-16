// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"fmt"
	"internal/coverage"
	"internal/coverage/decodecounter"
	"internal/coverage/encodecounter"
	"io"
	"os"
	"path/filepath"
	"testing"
)

type ctrVis struct {
	funcs []decodecounter.FuncPayload
}

func (v *ctrVis) VisitFuncs(f encodecounter.CounterVisitorFn) error {
	for _, fn := range v.funcs {
		if err := f(fn.PkgIdx, fn.FuncIdx, fn.Counters); err != nil {
			return err
		}
	}
	return nil
}

func mkfunc(p uint32, f uint32, c []uint32) decodecounter.FuncPayload {
	return decodecounter.FuncPayload{
		PkgIdx:   p,
		FuncIdx:  f,
		Counters: c,
	}
}

func TestCounterDataWriterReader(t *testing.T) {
	flavors := []coverage.CounterFlavor{
		coverage.CtrRaw,
		coverage.CtrULeb128,
	}

	isDead := func(fp decodecounter.FuncPayload) bool {
		for _, v := range fp.Counters {
			if v != 0 {
				return false
			}
		}
		return true
	}

	funcs := []decodecounter.FuncPayload{
		mkfunc(0, 0, []uint32{13, 14, 15}),
		mkfunc(0, 1, []uint32{16, 17}),
		mkfunc(1, 0, []uint32{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 976543, 7}),
	}
	writeVisitor := &ctrVis{funcs: funcs}

	for kf, flav := range flavors {

		t.Logf("testing flavor %d\n", flav)

		// Open a counter data file in preparation for emitting data.
		d := t.TempDir()
		cfpath := filepath.Join(d, fmt.Sprintf("covcounters.hash.0.%d", kf))
		of, err := os.OpenFile(cfpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			t.Fatalf("opening covcounters: %v", err)
		}

		// Perform the encode and write.
		cdfw := encodecounter.NewCoverageDataWriter(of, flav)
		if cdfw == nil {
			t.Fatalf("NewCoverageDataWriter failed")
		}
		finalHash := [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}
		args := map[string]string{"argc": "3", "argv0": "arg0", "argv1": "arg1", "argv2": "arg_________2"}
		if err := cdfw.Write(finalHash, args, writeVisitor); err != nil {
			t.Fatalf("counter file Write failed: %v", err)
		}
		if err := of.Close(); err != nil {
			t.Fatalf("closing covcounters: %v", err)
		}
		cdfw = nil

		// Decode the same file.
		var cdr *decodecounter.CounterDataReader
		inf, err := os.Open(cfpath)
		defer func() {
			if err := inf.Close(); err != nil {
				t.Fatalf("close failed with: %v", err)
			}
		}()

		if err != nil {
			t.Fatalf("reopening covcounters file: %v", err)
		}
		if cdr, err = decodecounter.NewCounterDataReader(cfpath, inf); err != nil {
			t.Fatalf("opening covcounters for read: %v", err)
		}
		decodedArgs := cdr.OsArgs()
		aWant := "[arg0 arg1 arg_________2]"
		aGot := fmt.Sprintf("%+v", decodedArgs)
		if aWant != aGot {
			t.Errorf("reading decoded args, got %s want %s", aGot, aWant)
		}
		for i := range funcs {
			if isDead(funcs[i]) {
				continue
			}
			var fp decodecounter.FuncPayload
			if ok, err := cdr.NextFunc(&fp); err != nil {
				t.Fatalf("reading func %d: %v", i, err)
			} else if !ok {
				t.Fatalf("reading func %d: bad return", i)
			}
			got := fmt.Sprintf("%+v", fp)
			want := fmt.Sprintf("%+v", funcs[i])
			if got != want {
				t.Errorf("cdr.NextFunc iter %d\ngot  %+v\nwant %+v", i, got, want)
			}
		}
		var dummy decodecounter.FuncPayload
		if ok, err := cdr.NextFunc(&dummy); err != nil {
			t.Fatalf("reading func after loop: %v", err)
		} else if ok {
			t.Fatalf("reading func after loop: expected EOF")
		}
	}
}

func TestCounterDataAppendSegment(t *testing.T) {
	d := t.TempDir()
	cfpath := filepath.Join(d, "covcounters.hash2.0")
	of, err := os.OpenFile(cfpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		t.Fatalf("opening covcounters: %v", err)
	}

	const numSegments = 2

	// Write a counter with with multiple segments.
	args := map[string]string{"argc": "1", "argv0": "prog.exe"}
	allfuncs := [][]decodecounter.FuncPayload{}
	ctrs := []uint32{}
	q := uint32(0)
	var cdfw *encodecounter.CoverageDataWriter
	for idx := 0; idx < numSegments; idx++ {
		args[fmt.Sprintf("seg%d", idx)] = "x"
		q += 7
		ctrs = append(ctrs, q)
		funcs := []decodecounter.FuncPayload{}
		for k := 0; k < idx+1; k++ {
			c := make([]uint32, len(ctrs))
			copy(c, ctrs)
			funcs = append(funcs, mkfunc(uint32(idx), uint32(k), c))
		}
		allfuncs = append(allfuncs, funcs)

		writeVisitor := &ctrVis{funcs: funcs}

		if idx == 0 {
			// Perform the encode and write.
			cdfw = encodecounter.NewCoverageDataWriter(of, coverage.CtrRaw)
			if cdfw == nil {
				t.Fatalf("NewCoverageDataWriter failed")
			}
			finalHash := [16]byte{1, 2}
			if err := cdfw.Write(finalHash, args, writeVisitor); err != nil {
				t.Fatalf("counter file Write failed: %v", err)
			}
		} else {
			if err := cdfw.AppendSegment(args, writeVisitor); err != nil {
				t.Fatalf("counter file AppendSegment failed: %v", err)
			}
		}
	}
	if err := of.Close(); err != nil {
		t.Fatalf("closing covcounters: %v", err)
	}

	// Read the result file.
	var cdr *decodecounter.CounterDataReader
	inf, err := os.Open(cfpath)
	defer func() {
		if err := inf.Close(); err != nil {
			t.Fatalf("close failed with: %v", err)
		}
	}()

	if err != nil {
		t.Fatalf("reopening covcounters file: %v", err)
	}
	if cdr, err = decodecounter.NewCounterDataReader(cfpath, inf); err != nil {
		t.Fatalf("opening covcounters for read: %v", err)
	}
	ns := cdr.NumSegments()
	if ns != numSegments {
		t.Fatalf("got %d segments want %d", ns, numSegments)
	}
	if len(allfuncs) != numSegments {
		t.Fatalf("expected %d got %d", numSegments, len(allfuncs))
	}

	for sidx := 0; sidx < int(ns); sidx++ {
		if off, err := inf.Seek(0, io.SeekCurrent); err != nil {
			t.Fatalf("Seek failed: %v", err)
		} else {
			t.Logf("sidx=%d off=%d\n", sidx, off)
		}

		if sidx != 0 {
			if ok, err := cdr.BeginNextSegment(); err != nil {
				t.Fatalf("BeginNextSegment failed: %v", err)
			} else if !ok {
				t.Fatalf("BeginNextSegment return %v on iter %d",
					ok, sidx)
			}
		}
		funcs := allfuncs[sidx]
		for i := range funcs {
			var fp decodecounter.FuncPayload
			if ok, err := cdr.NextFunc(&fp); err != nil {
				t.Fatalf("reading func %d: %v", i, err)
			} else if !ok {
				t.Fatalf("reading func %d: bad return", i)
			}
			got := fmt.Sprintf("%+v", fp)
			want := fmt.Sprintf("%+v", funcs[i])
			if got != want {
				t.Errorf("cdr.NextFunc iter %d\ngot  %+v\nwant %+v", i, got, want)
			}
		}
	}
}
