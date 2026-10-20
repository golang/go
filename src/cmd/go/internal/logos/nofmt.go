// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"slices"

	"cmd/go/internal/base"
)

// dropFmt removes the fmt subcommand in logos mode. No logos fmt
// command exists: formatting never rewrites .lgo files, and logos next
// is the only rewriting command.
func dropFmt(cmd *base.Command) {
	cmd.Commands = slices.DeleteFunc(cmd.Commands, func(sub *base.Command) bool {
		return sub.Name() == "fmt"
	})
}
