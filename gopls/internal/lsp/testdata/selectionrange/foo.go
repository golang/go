package foo

import "time"

func Bar(x, y int, t time.Time) int {
	zs := []int{1, 2, 3} //@selectionrange("1")

	for _, z := range zs {
		x = x + z + y + zs[1] //@selectionrange("1")
	}

	return x + y //@selectionrange("+")
}
