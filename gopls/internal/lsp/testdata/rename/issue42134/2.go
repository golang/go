package issue42134

import "fmt"

func _() {
	// minNumber is a min number.
	// Second line.
	minNumber := min(1, 2)
	fmt.Println(minNumber) //@rename("minNumber", "res")
}

func min(a, b int) int { return a }
