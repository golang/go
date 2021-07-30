// Package another has another type.
package another

import (
	other "golang.org/x/tools/internal/lsp/references/other"
)

func _() {
	xes := other.GetXes()
	for _, x := range xes { //@mark(defX, "x")
		_ = x.Y //@mark(useX, "x"),mark(anotherXY, "Y"),refs("Y", typeXY, anotherXY, GetXesY),refs(".", defX, useX),refs("x", defX, useX)
	}
}
