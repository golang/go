// Package another has another type.
package another

import (
	other "golang.org/x/tools/internal/lsp/references/other"
)

func _() {
	xes := other.GetXes()
	for _, x := range xes {
		_ = x.Y //@mark(anotherXY, "Y"),refs("Y", typeXY, anotherXY, GetXesY)
	}
}
