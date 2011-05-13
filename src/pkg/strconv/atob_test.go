// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"os"
	. "strconv"
	"testing"
)

type atobTest struct {
	in  string
	out bool
	err os.Error
}

var atobtests = []atobTest{
	{"", false, os.EINVAL},
	{"asdf", false, os.EINVAL},
	{"0", false, nil},
	{"f", false, nil},
	{"F", false, nil},
	{"FALSE", false, nil},
	{"false", false, nil},
	{"False", false, nil},
	{"1", true, nil},
	{"t", true, nil},
	{"T", true, nil},
	{"TRUE", true, nil},
	{"true", true, nil},
	{"True", true, nil},
}

func TestAtob(t *testing.T) {
	for _, test := range atobtests {
		b, e := Atob(test.in)
		if test.err != nil {
			// expect an error
			if e == nil {
				t.Errorf("%s: expected %s but got nil", test.in, test.err)
			} else {
				// NumError assertion must succeed; it's the only thing we return.
				if test.err != e.(*NumError).Error {
					t.Errorf("%s: expected %s but got %s", test.in, test.err, e)
				}
			}
		} else {
			if e != nil {
				t.Errorf("%s: expected no error but got %s", test.in, e)
			}
			if b != test.out {
				t.Errorf("%s: expected %t but got %t", test.in, test.out, b)
			}
		}
	}
}
