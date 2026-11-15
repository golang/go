// asmcheck

package main

type link struct{ next *link }

//go:noinline
func loadLink(p **link) *link {
	// amd64:"MOVQ \\(AX\\), AX" -"CALL" -".*writeBarrier" -"LOCK" -"XCHG"
	// arm64:"MOVD \\(R0\\), R0" -"CALL" -".*writeBarrier" -"LDAR" -"DMB"
	return *p
}

func main() {}
