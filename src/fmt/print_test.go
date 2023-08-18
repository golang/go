// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import "testing"

func Test_intFromArg(t *testing.T) {
	type myint int
	type args struct {
		a      []any
		argNum int
	}
	type want struct {
		num       int
		isInt     bool
		newArgNum int
	}
	var tests = []struct {
		name string
		args
		want
	}{
		{"myint", args{[]any{myint(0)}, 0}, want{0, true, 1}},

		{"int", args{[]any{1}, 0}, want{1, true, 1}},
		{"int8", args{[]any{int8(2)}, 0}, want{2, true, 1}},
		{"int16", args{[]any{int16(3)}, 0}, want{3, true, 1}},
		{"int32", args{[]any{int32(4)}, 0}, want{4, true, 1}},
		{"int64", args{[]any{int64(5)}, 0}, want{5, true, 1}},

		{"uint", args{[]any{uint(6)}, 0}, want{6, true, 1}},
		{"uint8", args{[]any{uint8(7)}, 0}, want{7, true, 1}},
		{"uint16", args{[]any{uint16(8)}, 0}, want{8, true, 1}},
		{"uint32", args{[]any{uint32(9)}, 0}, want{9, true, 1}},
		{"uint64", args{[]any{uint64(10)}, 0}, want{10, true, 1}},

		{"toolarge", args{[]any{1e6 + 11}, 0}, want{0, false, 1}},
		{"float", args{[]any{12.0}, 0}, want{0, false, 1}},
		{"struct", args{[]any{struct{ int }{13}}, 0}, want{0, false, 1}},
		{"bool", args{[]any{false}, 0}, want{0, false, 1}},
	}
	for _, test := range tests {
		num, isInt, newArgNum := intFromArg(test.args.a, test.args.argNum)
		if num != test.want.num {
			t.Errorf("%s num - got: %d, want: %d", test.name, num, test.want.num)
		}
		if isInt != test.want.isInt {
			t.Errorf("%s isInt - got: %t, want: %t", test.name, isInt, test.want.isInt)
		}
		if newArgNum != test.want.newArgNum {
			t.Errorf("%s newArgNum - got: %d, want: %d", test.name, newArgNum, test.want.newArgNum)
		}
	}
}
