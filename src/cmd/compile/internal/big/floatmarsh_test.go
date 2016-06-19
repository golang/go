// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"encoding/json"
	"testing"
)

var floatVals = []string{
	"0",
	"1",
	"0.1",
	"2.71828",
	"1234567890",
	"3.14e1234",
	"3.14e-1234",
	"0.738957395793475734757349579759957975985497e100",
	"0.73895739579347546656564656573475734957975995797598589749859834759476745986795497e100",
	"inf",
	"Inf",
}

func TestFloatJSONEncoding(t *testing.T) {
	for _, test := range floatVals {
		for _, sign := range []string{"", "+", "-"} {
			for _, prec := range []uint{0, 1, 2, 10, 53, 64, 100, 1000} {
				x := sign + test
				var tx Float
				_, _, err := tx.SetPrec(prec).Parse(x, 0)
				if err != nil {
					t.Errorf("parsing of %s (prec = %d) failed (invalid test case): %v", x, prec, err)
					continue
				}
				b, err := json.Marshal(&tx)
				if err != nil {
					t.Errorf("marshaling of %v (prec = %d) failed: %v", &tx, prec, err)
					continue
				}
				var rx Float
				rx.SetPrec(prec)
				if err := json.Unmarshal(b, &rx); err != nil {
					t.Errorf("unmarshaling of %v (prec = %d) failed: %v", &tx, prec, err)
					continue
				}
				if rx.Cmp(&tx) != 0 {
					t.Errorf("JSON encoding of %v (prec = %d) failed: got %v want %v", &tx, prec, &rx, &tx)
				}
			}
		}
	}
}
