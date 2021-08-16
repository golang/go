// errorcheckoutput

package main

import "fmt"

// We are going to define 256 types T(n),
// such that T(n) embeds T(2n) and *T(2n+1).

func main() {
	fmt.Printf("// errorcheck\n\n")
	fmt.Printf("package p\n\n")
	fmt.Println(`import "unsafe"`)

	// Dump types.
	for n := 1; n < 256; n++ {
		writeStruct(n)
	}
	// Dump leaves
	for n := 256; n < 512; n++ {
		fmt.Printf("type T%d int\n", n)
	}

	fmt.Printf("var t T1\n")
	fmt.Printf("var p *T1\n")

	// Simple selectors
	for n := 2; n < 256; n++ {
		writeDot(n)
	}

	// Double selectors
	for n := 128; n < 256; n++ {
		writeDot(n/16, n)
	}

	// Triple selectors
	for n := 128; n < 256; n++ {
		writeDot(n/64, n/8, n)
	}
}

const structTpl = `
type T%d struct {
	A%d int
	T%d
	*T%d
}
`

func writeStruct(n int) {
	fmt.Printf(structTpl, n, n, 2*n, 2*n+1)
}

func writeDot(ns ...int) {
	for _, root := range []string{"t", "p"} {
		fmt.Printf("const _ = unsafe.Offsetof(%s", root)
		for _, n := range ns {
			fmt.Printf(".T%d", n)
		}
		// Does it involve an indirection?
		nlast := ns[len(ns)-1]
		nprev := 1
		if len(ns) > 1 {
			nprev = ns[len(ns)-2]
		}
		isIndirect := false
		for n := nlast / 2; n > nprev; n /= 2 {
			if n%2 == 1 {
				isIndirect = true
				break
			}
		}
		fmt.Print(")")
		if isIndirect {
			fmt.Print(` // ERROR "indirection|embedded via a pointer"`)
		}
		fmt.Print("\n")
	}
}
