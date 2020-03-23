// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocommand_test

import (
	"context"
	"testing"

	"golang.org/x/tools/internal/gocommand"
)

func TestGoVersion(t *testing.T) {
	inv := gocommand.Invocation{
		Verb: "version",
	}
	gocmdRunner := &gocommand.Runner{}
	if _, err := gocmdRunner.Run(context.Background(), inv); err != nil {
		t.Error(err)
	}
}
