package good //@diag("package", "")

import (
	"golang.org/x/tools/internal/lsp/types" //@item(types_import, "types", "\"golang.org/x/tools/internal/lsp/types\"", "package")
)

func random() int { //@item(good_random, "random()", "int", "func")
	y := 6 + 7
	return y
}

func random2(y int) int { //@item(good_random2, "random2(y int)", "int", "func"),item(good_y_param, "y", "int", "var")
	//@complete("", good_y_param, types_import, good_random, good_random2, good_stuff)
	var b types.Bob = &types.X{}
	if _, ok := b.(*types.X); ok { //@complete("X", Bob_interface, X_struct, Y_struct)
	}

	return y
}
