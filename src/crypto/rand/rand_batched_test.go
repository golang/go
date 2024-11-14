// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package rand

import (
	"bytes"
	"errors"
	prand "math/rand"
	"testing"
)

func TestBatched(t *testing.T) {
	fillBatched := batched(func { p ->
		for i := range p {
			p[i] = byte(i)
		}
		return nil
	}, 5)

	p := make([]byte, 13)
	if err := fillBatched(p); err != nil {
		t.Fatalf("batched function returned error: %s", err)
	}
	expected := []byte{0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2}
	if !bytes.Equal(expected, p) {
		t.Errorf("incorrect batch result: got %x, want %x", p, expected)
	}
}

func TestBatchedBuffering(t *testing.T) {
	backingStore := make([]byte, 1<<23)
	prand.Read(backingStore)
	backingMarker := backingStore[:]
	output := make([]byte, len(backingStore))
	outputMarker := output[:]

	fillBatched := batched(func { p ->
		n := copy(p, backingMarker)
		backingMarker = backingMarker[n:]
		return nil
	}, 731)

	for len(outputMarker) > 0 {
		max := 9200
		if max > len(outputMarker) {
			max = len(outputMarker)
		}
		howMuch := prand.Intn(max + 1)
		if err := fillBatched(outputMarker[:howMuch]); err != nil {
			t.Fatalf("batched function returned error: %s", err)
		}
		outputMarker = outputMarker[howMuch:]
	}
	if !bytes.Equal(backingStore, output) {
		t.Error("incorrect batch result")
	}
}

func TestBatchedError(t *testing.T) {
	b := batched(func { p -> errors.New("failure") }, 5)
	if b(make([]byte, 13)) == nil {
		t.Fatal("batched function should have returned an error")
	}
}

func TestBatchedEmpty(t *testing.T) {
	b := batched(func { p -> errors.New("failure") }, 5)
	if b(make([]byte, 0)) != nil {
		t.Fatal("empty slice should always return successful")
	}
}
