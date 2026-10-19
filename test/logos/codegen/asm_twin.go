// asmcheck

package codegen

func lshConst64x1(v int64) int64 {
	// amd64:"ADDQ" -"SHLQ"
	return v << 1
}
