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
	atobTest{"", false, os.EINVAL},
	atobTest{"asdf", false, os.EINVAL},
	atobTest{"0", false, nil},
	atobTest{"f", false, nil},
	atobTest{"F", false, nil},
	atobTest{"FALSE", false, nil},
	atobTest{"false", false, nil},
	atobTest{"1", true, nil},
	atobTest{"t", true, nil},
	atobTest{"T", true, nil},
	atobTest{"TRUE", true, nil},
	atobTest{"true", true, nil},
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
				t.Errorf("%s: expected no error but got %s", test.in, test.err, e)
			}
			if b != test.out {
				t.Errorf("%s: expected %t but got %t", test.in, test.out, b)
			}
		}
	}
}
