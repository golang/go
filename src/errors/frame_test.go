// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors_test

import (
	"bytes"
	"errors"
	"fmt"
	"math/big"
	"regexp"
	"strings"
	"testing"
)

func TestFrame(t *testing.T) {

	// Extra line
	got := fmt.Sprintf("%+v", errors.New("Test"))
	got = got[strings.Index(got, "Test"):]
	const want = "^Test:" +
		"\n    errors_test.TestFrame" +
		"\n        .*/errors/frame_test.go:20$"
	ok, err := regexp.MatchString(want, got)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Errorf("\n got %v;\nwant %v", got, want)
	}
}

type myType struct{}

func (myType) Format(s fmt.State, v rune) {
	s.Write(bytes.Repeat([]byte("Hi! "), 10))
}

func BenchmarkNew(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = errors.New("new error")
	}
}

func BenchmarkErrorf(b *testing.B) {
	err := errors.New("foo")
	args := func(a ...interface{}) []interface{} { return a }
	benchCases := []struct {
		name   string
		format string
		args   []interface{}
	}{
		{"no_format", "msg: %v", args(err)},
		{"with_format", "failed %d times: %v", args(5, err)},
		{"method: mytype", "pi %s %v: %v", args("myfile.go", myType{}, err)},
		{"method: number", "pi %s %d: %v", args("myfile.go", big.NewInt(5), err)},
	}
	for _, bc := range benchCases {
		b.Run(bc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = fmt.Errorf(bc.format, bc.args...)
			}
		})
	}
}
