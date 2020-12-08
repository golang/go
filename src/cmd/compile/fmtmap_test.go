// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the knownFormats map which records the valid
// formats for a given type. The valid formats must correspond to
// supported compiler formats implemented in fmt.go, or whatever
// other format verbs are implemented for the given type. The map may
// also be used to change the use of a format verb across all compiler
// sources automatically (for instance, if the implementation of fmt.go
// changes), by using the -r option together with the new formats in the
// map. To generate this file automatically from the existing source,
// run: go test -run Formats -u.
//
// See the package comment in fmt_test.go for additional information.

package main_test

// knownFormats entries are of the form "typename format" -> "newformat".
// An absent entry means that the format is not recognized as valid.
// An empty new format means that the format should remain unchanged.
var knownFormats = map[string]string{
	"*bytes.Buffer %s":                             "",
	"*cmd/compile/internal/ssa.Block %s":           "",
	"*cmd/compile/internal/ssa.Func %s":            "",
	"*cmd/compile/internal/ssa.Register %s":        "",
	"*cmd/compile/internal/ssa.Value %s":           "",
	"*cmd/compile/internal/types.Sym %+v":          "",
	"*cmd/compile/internal/types.Sym %S":           "",
	"*cmd/compile/internal/types.Type %+v":         "",
	"*cmd/compile/internal/types.Type %-S":         "",
	"*cmd/compile/internal/types.Type %L":          "",
	"*cmd/compile/internal/types.Type %S":          "",
	"*cmd/compile/internal/types.Type %s":          "",
	"*math/big.Float %f":                           "",
	"*math/big.Int %s":                             "",
	"[]cmd/compile/internal/syntax.token %s":       "",
	"cmd/compile/internal/arm.shift %d":            "",
	"cmd/compile/internal/gc.initKind %d":          "",
	"cmd/compile/internal/ir.Class %d":             "",
	"cmd/compile/internal/ir.Node %+v":             "",
	"cmd/compile/internal/ir.Node %L":              "",
	"cmd/compile/internal/ir.Nodes %+v":            "",
	"cmd/compile/internal/ir.Nodes %.v":            "",
	"cmd/compile/internal/ir.Op %+v":               "",
	"cmd/compile/internal/ssa.Aux %#v":             "",
	"cmd/compile/internal/ssa.Aux %q":              "",
	"cmd/compile/internal/ssa.Aux %s":              "",
	"cmd/compile/internal/ssa.BranchPrediction %d": "",
	"cmd/compile/internal/ssa.ID %d":               "",
	"cmd/compile/internal/ssa.LocalSlot %s":        "",
	"cmd/compile/internal/ssa.Location %s":         "",
	"cmd/compile/internal/ssa.Op %s":               "",
	"cmd/compile/internal/ssa.ValAndOff %s":        "",
	"cmd/compile/internal/ssa.flagConstant %s":     "",
	"cmd/compile/internal/ssa.rbrank %d":           "",
	"cmd/compile/internal/ssa.regMask %d":          "",
	"cmd/compile/internal/ssa.register %d":         "",
	"cmd/compile/internal/ssa.relation %s":         "",
	"cmd/compile/internal/syntax.Error %q":         "",
	"cmd/compile/internal/syntax.Expr %#v":         "",
	"cmd/compile/internal/syntax.LitKind %d":       "",
	"cmd/compile/internal/syntax.Operator %s":      "",
	"cmd/compile/internal/syntax.Pos %s":           "",
	"cmd/compile/internal/syntax.position %s":      "",
	"cmd/compile/internal/syntax.token %q":         "",
	"cmd/compile/internal/syntax.token %s":         "",
	"cmd/compile/internal/types.Kind %d":           "",
	"cmd/compile/internal/types.Kind %s":           "",
	"go/constant.Value %#v":                        "",
	"math/big.Accuracy %s":                         "",
	"reflect.Type %s":                              "",
	"time.Duration %d":                             "",
}
