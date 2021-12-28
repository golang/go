// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"cmd/go/internal/modget"
)

var cmdUpdate = modget.CmdGet

func init() {
	cmdUpdate.UsageLine = "go mod update [-t] [-u] [-v] [build flags] [packages]"
	cmdUpdate.Run = modget.RunGet
}
