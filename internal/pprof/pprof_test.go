// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof_test

import (
	"bytes"
	"compress/gzip"
	"io"
	"log"
	"os"
	"testing"
	"time"

	"golang.org/x/tools/internal/pprof"
)

func TestTotalTime(t *testing.T) {
	// $ go tool pprof testdata/sample.pprof <&- 2>&1 | grep Total
	// Duration: 11.10s, Total samples = 27.59s (248.65%)
	const (
		filename = "testdata/sample.pprof"
		want     = time.Duration(27590003550)
	)

	profGz, err := os.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	rd, err := gzip.NewReader(bytes.NewReader(profGz))
	if err != nil {
		t.Fatal(err)
	}
	payload, err := io.ReadAll(rd)
	if err != nil {
		t.Fatal(err)
	}
	got, err := pprof.TotalTime(payload)
	if err != nil {
		log.Fatal(err)
	}
	if got != want {
		t.Fatalf("TotalTime(%q): got %v (%d), want %v (%d)", filename, got, got, want, want)
	}
}
