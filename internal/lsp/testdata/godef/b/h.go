package b

import . "golang.org/x/tools/internal/lsp/godef/a"

func _() {
	// variable of type a.A
	var _ A //@mark(AVariable, "_"),hover("_", AVariable)

	AStuff() //@hover("AStuff", AStuff)
}
