// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate_test

import (
	"exp/locale/collate"
	"fmt"
	"testing"
)

func ExampleCollator_Strings() {
	c := collate.New("root")
	strings := []string{
		"ad",
		"äb",
		"ac",
	}
	c.Strings(strings)
	fmt.Println(strings)
	// Output: [äb ac ad]
}

type sorter []string

func (s sorter) Len() int {
	return len(s)
}

func (s sorter) Swap(i, j int) {
	s[j], s[i] = s[i], s[j]
}

func (s sorter) Bytes(i int) []byte {
	return []byte(s[i])
}

func TestSort(t *testing.T) {
	c := collate.New("en")
	strings := []string{
		"bcd",
		"abc",
		"ddd",
	}
	c.Sort(sorter(strings))
	res := fmt.Sprint(strings)
	want := "[abc bcd ddd]"
	if res != want {
		t.Errorf("found %s; want %s", res, want)
	}
}
