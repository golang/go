// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package saferio

import (
	"bytes"
	"testing"
)

func TestReadData(t *testing.T) {
	const count = 100
	input := bytes.Repeat([]byte{'a'}, count)

	t.Run("small", func(t *testing.T) {
		got, err := ReadData(bytes.NewReader(input), count)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, input) {
			t.Errorf("got %v, want %v", got, input)
		}
	})

	t.Run("large", func(t *testing.T) {
		_, err := ReadData(bytes.NewReader(input), 10<<30)
		if err == nil {
			t.Error("large read succeeded unexpectedly")
		}
	})

	t.Run("maxint", func(t *testing.T) {
		_, err := ReadData(bytes.NewReader(input), 1<<62)
		if err == nil {
			t.Error("large read succeeded unexpectedly")
		}
	})
}
