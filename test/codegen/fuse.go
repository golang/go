// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Notes:
// - these examples use channels to provide a source of
//   unknown values that cannot be optimized away
// - these examples use for loops to force branches
//   backward (predicted taken)

// ---------------------------------- //
// signed integer range (conjunction) //
// ---------------------------------- //

func si1c(c <-chan int64) {
	// amd64:"CMPQ\t.+, [$]256"
	// s390x:"CLGIJ\t[$]12, R[0-9]+, [$]255"
	for x := <-c; x >= 0 && x < 256; x = <-c {
	}
}

func si2c(c <-chan int32) {
	// amd64:"CMPL\t.+, [$]256"
	// s390x:"CLIJ\t[$]12, R[0-9]+, [$]255"
	for x := <-c; x >= 0 && x < 256; x = <-c {
	}
}

func si3c(c <-chan int16) {
	// amd64:"CMPW\t.+, [$]256"
	// s390x:"CLIJ\t[$]12, R[0-9]+, [$]255"
	for x := <-c; x >= 0 && x < 256; x = <-c {
	}
}

func si4c(c <-chan int8) {
	// amd64:"CMPB\t.+, [$]10"
	// s390x:"CLIJ\t[$]4, R[0-9]+, [$]10"
	for x := <-c; x >= 0 && x < 10; x = <-c {
	}
}

func si5c(c <-chan int64) {
	// amd64:"CMPQ\t.+, [$]251","ADDQ\t[$]-5,"
	// s390x:"CLGIJ\t[$]4, R[0-9]+, [$]251","ADD\t[$]-5,"
	for x := <-c; x < 256 && x > 4; x = <-c {
	}
}

func si6c(c <-chan int32) {
	// amd64:"CMPL\t.+, [$]255","DECL\t"
	// s390x:"CLIJ\t[$]12, R[0-9]+, [$]255","ADDW\t[$]-1,"
	for x := <-c; x > 0 && x <= 256; x = <-c {
	}
}

func si7c(c <-chan int16) {
	// amd64:"CMPW\t.+, [$]60","ADDL\t[$]10,"
	// s390x:"CLIJ\t[$]12, R[0-9]+, [$]60","ADDW\t[$]10,"
	for x := <-c; x >= -10 && x <= 50; x = <-c {
	}
}

func si8c(c <-chan int8) {
	// amd64:"CMPB\t.+, [$]126","ADDL\t[$]126,"
	// s390x:"CLIJ\t[$]4, R[0-9]+, [$]126","ADDW\t[$]126,"
	for x := <-c; x >= -126 && x < 0; x = <-c {
	}
}

// ---------------------------------- //
// signed integer range (disjunction) //
// ---------------------------------- //

func si1d(c <-chan int64) {
	// amd64:"CMPQ\t.+, [$]256"
	// s390x:"CLGIJ\t[$]2, R[0-9]+, [$]255"
	for x := <-c; x < 0 || x >= 256; x = <-c {
	}
}

func si2d(c <-chan int32) {
	// amd64:"CMPL\t.+, [$]256"
	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]255"
	for x := <-c; x < 0 || x >= 256; x = <-c {
	}
}

func si3d(c <-chan int16) {
	// amd64:"CMPW\t.+, [$]256"
	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]255"
	for x := <-c; x < 0 || x >= 256; x = <-c {
	}
}

func si4d(c <-chan int8) {
	// amd64:"CMPB\t.+, [$]10"
	// s390x:"CLIJ\t[$]10, R[0-9]+, [$]10"
	for x := <-c; x < 0 || x >= 10; x = <-c {
	}
}

func si5d(c <-chan int64) {
	// amd64:"CMPQ\t.+, [$]251","ADDQ\t[$]-5,"
	// s390x:"CLGIJ\t[$]10, R[0-9]+, [$]251","ADD\t[$]-5,"
	for x := <-c; x >= 256 || x <= 4; x = <-c {
	}
}

func si6d(c <-chan int32) {
	// amd64:"CMPL\t.+, [$]255","DECL\t"
	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]255","ADDW\t[$]-1,"
	for x := <-c; x <= 0 || x > 256; x = <-c {
	}
}

func si7d(c <-chan int16) {
	// amd64:"CMPW\t.+, [$]60","ADDL\t[$]10,"
	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]60","ADDW\t[$]10,"
	for x := <-c; x < -10 || x > 50; x = <-c {
	}
}

func si8d(c <-chan int8) {
	// amd64:"CMPB\t.+, [$]126","ADDL\t[$]126,"
	// s390x:"CLIJ\t[$]10, R[0-9]+, [$]126","ADDW\t[$]126,"
	for x := <-c; x < -126 || x >= 0; x = <-c {
	}
}

// ------------------------------------ //
// unsigned integer range (conjunction) //
// ------------------------------------ //

func ui1c(c <-chan uint64) {
	// amd64:"CMPQ\t.+, [$]251","ADDQ\t[$]-5,"
	// s390x:"CLGIJ\t[$]4, R[0-9]+, [$]251","ADD\t[$]-5,"
	for x := <-c; x < 256 && x > 4; x = <-c {
	}
}

func ui2c(c <-chan uint32) {
	// amd64:"CMPL\t.+, [$]255","DECL\t"
	// s390x:"CLIJ\t[$]12, R[0-9]+, [$]255","ADDW\t[$]-1,"
	for x := <-c; x > 0 && x <= 256; x = <-c {
	}
}

func ui3c(c <-chan uint16) {
	// amd64:"CMPW\t.+, [$]40","ADDL\t[$]-10,"
	// s390x:"CLIJ\t[$]12, R[0-9]+, [$]40","ADDW\t[$]-10,"
	for x := <-c; x >= 10 && x <= 50; x = <-c {
	}
}

func ui4c(c <-chan uint8) {
	// amd64:"CMPB\t.+, [$]2","ADDL\t[$]-126,"
	// s390x:"CLIJ\t[$]4, R[0-9]+, [$]2","ADDW\t[$]-126,"
	for x := <-c; x >= 126 && x < 128; x = <-c {
	}
}

// ------------------------------------ //
// unsigned integer range (disjunction) //
// ------------------------------------ //

func ui1d(c <-chan uint64) {
	// amd64:"CMPQ\t.+, [$]251","ADDQ\t[$]-5,"
	// s390x:"CLGIJ\t[$]10, R[0-9]+, [$]251","ADD\t[$]-5,"
	for x := <-c; x >= 256 || x <= 4; x = <-c {
	}
}

func ui2d(c <-chan uint32) {
	// amd64:"CMPL\t.+, [$]254","ADDL\t[$]-2,"
	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]254","ADDW\t[$]-2,"
	for x := <-c; x <= 1 || x > 256; x = <-c {
	}
}

func ui3d(c <-chan uint16) {
	// amd64:"CMPW\t.+, [$]40","ADDL\t[$]-10,"
	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]40","ADDW\t[$]-10,"
	for x := <-c; x < 10 || x > 50; x = <-c {
	}
}

func ui4d(c <-chan uint8) {
	// amd64:"CMPB\t.+, [$]2","ADDL\t[$]-126,"
	// s390x:"CLIJ\t[$]10, R[0-9]+, [$]2","ADDW\t[$]-126,"
	for x := <-c; x < 126 || x >= 128; x = <-c {
	}
}
