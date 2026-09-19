// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/testimporter"
	"cmd/compile/internal/types2"
)

var imp = testimporter.NewImporter()

func defaultImporter() types2.Importer {
	return imp
}
