// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pgo

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// equal returns an error if got and want are not equal.
func equal(got, want *Profile) error {
	if got.TotalWeight != want.TotalWeight {
		return fmt.Errorf("got.TotalWeight %d != want.TotalWeight %d", got.TotalWeight, want.TotalWeight)
	}
	if !reflect.DeepEqual(got.NamedEdgeMap.ByWeight, want.NamedEdgeMap.ByWeight) {
		return fmt.Errorf("got.NamedEdgeMap.ByWeight != want.NamedEdgeMap.ByWeight\ngot = %+v\nwant = %+v", got.NamedEdgeMap.ByWeight, want.NamedEdgeMap.ByWeight)
	}
	if !reflect.DeepEqual(got.NamedEdgeMap.Weight, want.NamedEdgeMap.Weight) {
		return fmt.Errorf("got.NamedEdgeMap.Weight != want.NamedEdgeMap.Weight\ngot = %+v\nwant = %+v", got.NamedEdgeMap.Weight, want.NamedEdgeMap.Weight)
	}

	return nil
}

func testRoundTrip(t *testing.T, d *Profile) []byte {
	var buf bytes.Buffer
	n, err := d.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo got err %v want nil", err)
	}
	if n != int64(buf.Len()) {
		t.Errorf("WriteTo got n %d want %d", n, int64(buf.Len()))
	}

	b := buf.Bytes()

	got, err := FromSerialized(&buf)
	if err != nil {
		t.Fatalf("processSerialized got err %v want nil", err)
	}
	if err := equal(got, d); err != nil {
		t.Errorf("processSerialized output does not match input: %v", err)
	}

	return b
}

func TestEmpty(t *testing.T) {
	d := emptyProfile()
	b := testRoundTrip(t, d)

	// Contents should consist of only a header.
	if string(b) != serializationHeader {
		t.Errorf("WriteTo got %q want %q", string(b), serializationHeader)
	}
}

func TestRoundTrip(t *testing.T) {
	d := &Profile{
		TotalWeight: 3,
		NamedEdgeMap: NamedEdgeMap{
			ByWeight: []NamedCallEdge{
				{
					CallerName: "a",
					CalleeName: "b",
					CallSiteOffset: 14,
				},
				{
					CallerName: "c",
					CalleeName: "d",
					CallSiteOffset: 15,
				},
			},
			Weight: map[NamedCallEdge]int64{
				{
					CallerName: "a",
					CalleeName: "b",
					CallSiteOffset: 14,
				}: 2,
				{
					CallerName: "c",
					CalleeName: "d",
					CallSiteOffset: 15,
				}: 1,
			},
		},
	}

	testRoundTrip(t, d)
}

func constructFuzzProfile(t *testing.T, b []byte) *Profile {
	// The fuzzer can't construct an arbitrary structure, so instead we
	// consume bytes from b to act as our edge data.
	r := bytes.NewReader(b)
	consumeString := func() (string, bool) {
		// First byte: how many bytes to read for this string? We only
		// use a byte to avoid making humongous strings.
		length, err := r.ReadByte()
		if err != nil {
			return "", false
		}
		if length == 0 {
			return "", false
		}

		b := make([]byte, length)
		_, err = r.Read(b)
		if err != nil {
			return "", false
		}

		return string(b), true
	}
	consumeInt64 := func() (int64, bool) {
		b := make([]byte, 8)
		_, err := r.Read(b)
		if err != nil {
			return 0, false
		}

		return int64(binary.LittleEndian.Uint64(b)), true
	}

	d := emptyProfile()

	for {
		caller, ok := consumeString()
		if !ok {
			break
		}
		if strings.ContainsAny(caller, " \r\n") {
			t.Skip("caller contains space or newline")
		}

		callee, ok := consumeString()
		if !ok {
			break
		}
		if strings.ContainsAny(callee, " \r\n") {
			t.Skip("callee contains space or newline")
		}

		line, ok := consumeInt64()
		if !ok {
			break
		}
		weight, ok := consumeInt64()
		if !ok {
			break
		}

		edge := NamedCallEdge{
			CallerName: caller,
			CalleeName: callee,
			CallSiteOffset: int(line),
		}

		if _, ok := d.NamedEdgeMap.Weight[edge]; ok {
			t.Skip("duplicate edge")
		}

		d.NamedEdgeMap.Weight[edge] = weight
		d.TotalWeight += weight
	}

	byWeight := make([]NamedCallEdge, 0, len(d.NamedEdgeMap.Weight))
	for namedEdge := range d.NamedEdgeMap.Weight {
		byWeight = append(byWeight, namedEdge)
	}
	sortByWeight(byWeight, d.NamedEdgeMap.Weight)
	d.NamedEdgeMap.ByWeight = byWeight

	return d
}

func FuzzRoundTrip(f *testing.F) {
	f.Add([]byte("")) // empty profile

	f.Fuzz(func(t *testing.T, b []byte) {
		d := constructFuzzProfile(t, b)
		testRoundTrip(t, d)
	})
}
