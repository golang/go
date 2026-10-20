// asmcheck

package main

// M19-S6 attack (SPEC Sec 8.4, T-301.2, T-PERF): under greentea a pointer
// load is one plain move. The collector never moves objects, so the load
// carries no write-barrier flag test, no barrier call, and no locked or
// ordered instruction.

type link struct{ next *link }

//go:noinline
func loadLink(p **link) *link {
	// amd64:"MOVQ \\(AX\\), AX" -"CALL" -"writeBarrier" -"LOCK" -"XCHG"
	// arm64:"MOVD \\(R0\\), R0" -"CALL" -"writeBarrier" -"LDAR" -"DMB"
	return *p
}

func main() {}
