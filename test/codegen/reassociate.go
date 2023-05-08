// asmcheck

package codegen

// reassociateAddition expects very specific sequence of registers
// of the form:
// R2 += R3
// R1 += R0
// R1 += R2
func reassociateAddition(a, b, c, d int) int {
	// arm64:`ADD\tR2,\sR3,\sR2`
	x := b + a
	// arm64:`ADD\tR0,\sR1,\sR1`
	y := x + c
	// arm64:`ADD\tR1,\sR2,\sR0`
	z := y + d
	return z
}