//go:build go1.18
// +build go1.18

package inlayHint //@inlayHint("package")

func main() {
	ints := map[string]int64{
		"first":  34,
		"second": 12,
	}

	floats := map[string]float64{
		"first":  35.98,
		"second": 26.99,
	}

	SumIntsOrFloats[string, int64](ints)
	SumIntsOrFloats[string, float64](floats)

	SumIntsOrFloats(ints)
	SumIntsOrFloats(floats)

	SumNumbers(ints)
	SumNumbers(floats)
}

type Number interface {
	int64 | float64
}

func SumIntsOrFloats[K comparable, V int64 | float64](m map[K]V) V {
	var s V
	for _, v := range m {
		s += v
	}
	return s
}

func SumNumbers[K comparable, V Number](m map[K]V) V {
	var s V
	for _, v := range m {
		s += v
	}
	return s
}
