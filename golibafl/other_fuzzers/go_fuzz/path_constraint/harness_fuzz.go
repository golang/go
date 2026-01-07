//go:build gofuzz

package harness

func FuzzMe(input []byte) int {
	if harness(input) {
		panic("crash")
		//t.Fatalf("Found input: %s", input)
	}
	return 0
}
