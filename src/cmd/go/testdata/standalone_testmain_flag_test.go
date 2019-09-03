// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package standalone_testmain_flag_test

import (
	"flag"
	"fmt"
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	// A TestMain should be able to access testing flags if it calls
	// flag.Parse without needing to use testing.Init.
	flag.Parse()
	found := false
	flag.VisitAll(func(f *flag.Flag) {
		if f.Name == "test.count" {
			found = true
		}
	})
	if !found {
		fmt.Println("testing flags not registered")
		os.Exit(1)
	}
	os.Exit(m.Run())
}
