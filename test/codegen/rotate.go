// asmcheck

package codegen

import "math"

func rot32(x uint32) uint32 {
	var a uint32
	a += x<<7 | x>>25 // amd64:"ROLL.*[$]7"  arm:"MOVW.*@>25"
	a += x<<8 + x>>24 // amd64:`ROLL.*\$8`   arm:"MOVW.*@>24"
	a += x<<9 ^ x>>23 // amd64:"ROLL.*\\$9"  arm:"MOVW.*@>23"
	return a
}

func rot64(x uint64) uint64 {
	var a uint64
	a += x<<7 | x>>57 // amd64:"ROL"
	a += x<<8 + x>>56 // amd64:"ROL"
	a += x<<9 ^ x>>55 // amd64:"ROL"
	return a
}

func copysign(a, b float64) float64 {
	// amd64:"SHLQ\t[$]1","SHRQ\t[$]1","SHRQ\t[$]63","SHLQ\t[$]63","ORQ"
	// ppc64le:"FCPSGN" s390x:"CPSDR",-"MOVD"
	return math.Copysign(a, b)
}
