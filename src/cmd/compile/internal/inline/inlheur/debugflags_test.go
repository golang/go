// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"testing"
)

func TestInlScoreAdjFlagParse(t *testing.T) {
	scenarios := []struct {
		value string
		expok bool
	}{
		{
			value: "returnFeedsConcreteToInterfaceCallAdj:9",
			expok: true,
		},
		{
			value: "panicPathAdj:-1/initFuncAdj:9",
			expok: true,
		},
		{
			value: "",
			expok: false,
		},
		{
			value: "nonsenseAdj:10",
			expok: false,
		},
		{
			value: "inLoopAdj:",
			expok: false,
		},
		{
			value: "inLoopAdj:10:10",
			expok: false,
		},
		{
			value: "inLoopAdj:blah",
			expok: false,
		},
		{
			value: "/",
			expok: false,
		},
	}

	for _, scenario := range scenarios {
		err := parseScoreAdj(scenario.value)
		t.Logf("for value=%q err is %v\n", scenario.value, err)
		if scenario.expok {
			if err != nil {
				t.Errorf("expected parseScoreAdj(%s) ok, got err %v",
					scenario.value, err)
			}
		} else {
			if err == nil {
				t.Errorf("expected parseScoreAdj(%s) failure, got success",
					scenario.value)
			}
		}
	}
}
