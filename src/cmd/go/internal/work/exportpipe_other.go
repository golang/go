// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cmd_go_bootstrap || (!unix && !windows)

package work

import "errors"

const earlyExportSupported = false

type exportPipe struct{}

func newExportPipe(objdir string) (*exportPipe, error) {
	return nil, errors.ErrUnsupported
}

func (ep *exportPipe) name() string { return "" }
func (ep *exportPipe) wait() error  { return errors.ErrUnsupported }
func (ep *exportPipe) close()       {}
