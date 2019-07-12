// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/token"
)

// modFile holds all of the information we know about a mod file.
type modFile struct {
	fileBase
}

func (*modFile) GetToken(context.Context) (*token.File, error) {
	return nil, fmt.Errorf("GetToken: not implemented")
}

func (*modFile) setContent(content []byte) {}
func (*modFile) filename() string          { return "" }
func (*modFile) isActive() bool            { return false }
