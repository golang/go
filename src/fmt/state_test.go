// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"fmt"
	"testing"
)

type testState struct {
	width   int
	widthOK bool
	prec    int
	precOK  bool
	flag    map[int]bool
}

var _ fmt.State = testState{}

func (s testState) Write(b []byte) (n int, err error) {
	panic("unimplemented")
}

func (s testState) Width() (wid int, ok bool) {
	return s.width, s.widthOK
}

func (s testState) Precision() (prec int, ok bool) {
	return s.prec, s.precOK
}

func (s testState) Flag(c int) bool {
	return s.flag[c]
}

const NO = -1000

func mkState(w, p int, flags string) testState {
	s := testState{}
	if w != NO {
		s.width = w
		s.widthOK = true
	}
	if p != NO {
		s.prec = p
		s.precOK = true
	}
	s.flag = make(map[int]bool)
	for _, c := range flags {
		s.flag[int(c)] = true
	}
	return s
}

func TestFormatString(t *testing.T) {
	var tests = []struct {
		width, prec int
		flags       string
		result      string
	}{
		{NO, NO, "", "%x"},
		{NO, 3, "", "%.3x"},
		{3, NO, "", "%3x"},
		{7, 3, "", "%7.3x"},
		{NO, NO, " +-#0", "% +-#0x"},
		{7, 3, "+", "%+7.3x"},
		{7, -3, "-", "%-7.-3x"},
		{7, 3, " ", "% 7.3x"},
		{7, 3, "#", "%#7.3x"},
		{7, 3, "0", "%07.3x"},
	}
	for _, test := range tests {
		got := fmt.FormatString(mkState(test.width, test.prec, test.flags), 'x')
		if got != test.result {
			t.Errorf("%v: got %s", test, got)
		}
	}
}
