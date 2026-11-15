// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/internal/goobj"
)

func TestWideParamMutatedRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	in := &facts.Facts{Fns: []facts.FnFact{{
		Object:       "F",
		ParamMutated: "80000000000000000000000000000000000000001",
	}}}
	out := string(facts.Encode(fp, in))
	if !strings.Contains(out, "fn F ParamMutated=80000000000000000000000000000000000000001\n") {
		t.Fatalf("wide ParamMutated not canonical:\n%s", out)
	}
	back, err := facts.Decode([]byte(out))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if len(back.Fns) != 1 || back.Fns[0].ParamMutated != "80000000000000000000000000000000000000001" {
		t.Fatalf("wide mask lost: %+v", back.Fns[0])
	}
}
