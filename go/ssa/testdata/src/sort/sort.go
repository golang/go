package sort

func Strings(x []string)
func Ints(x []int)
func Float64s(x []float64)

func Sort(data Interface)

type Interface interface {
	Len() int
	Less(i, j int) bool
	Swap(i, j int)
}
