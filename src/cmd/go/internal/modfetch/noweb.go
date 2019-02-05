// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cmd_go_bootstrap

package modfetch

import (
	"fmt"
	"io"
)

func webGetGoGet(url string, body *io.ReadCloser) error {
	return fmt.Errorf("no network in go_bootstrap")
}

func webGetBytes(url string, body *[]byte) error {
	return fmt.Errorf("no network in go_bootstrap")
}

func webGetBody(url string, body *io.ReadCloser) error {
	return fmt.Errorf("no network in go_bootstrap")
}
