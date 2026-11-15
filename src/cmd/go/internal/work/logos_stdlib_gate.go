// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"strings"

	"cmd/go/internal/load"
	"cmd/go/internal/search"
)

const (
	logosVariableMakeFlag = "-d=variablemakethreshold="
	logosDebugFlagPrefix  = "-d="
	logosDebugKeySep      = ","
)

func logosForcedGcflags(p *load.Package) []string {
	if !logosStandardProvenance(p) {
		return forcedGcflags
	}
	out := make([]string, 0, len(forcedGcflags))
	for _, f := range forcedGcflags {
		out = append(out, logosDropVariableMake(f))
	}
	return out
}

func logosStandardProvenance(p *load.Package) bool {
	if p.Standard {
		return true
	}
	return p.ForTest != "" && search.IsStandardImportPath(p.ForTest)
}

func logosDropVariableMake(flag string) string {
	value, ok := strings.CutPrefix(flag, logosDebugFlagPrefix)
	if !ok {
		return flag
	}
	keys := strings.Split(value, logosDebugKeySep)
	kept := make([]string, 0, len(keys))
	for _, k := range keys {
		if _, found := strings.CutPrefix(k, strings.TrimPrefix(logosVariableMakeFlag, logosDebugFlagPrefix)); found {
			continue
		}
		kept = append(kept, k)
	}
	if len(kept) == len(keys) {
		return flag
	}
	if len(kept) == 0 {
		return ""
	}
	return logosDebugFlagPrefix + strings.Join(kept, logosDebugKeySep)
}
