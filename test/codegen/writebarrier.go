// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func combine2string(p *[2]string, a, b string) {
	// amd64:`.*runtime[.]gcWriteBarrier4\(SB\)`
	// arm64:`.*runtime[.]gcWriteBarrier4\(SB\)`
	p[0] = a
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[1] = b
}

func combine4string(p *[4]string, a, b, c, d string) {
	// amd64:`.*runtime[.]gcWriteBarrier8\(SB\)`
	// arm64:`.*runtime[.]gcWriteBarrier8\(SB\)`
	p[0] = a
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[1] = b
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[2] = c
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[3] = d
}

func combine2slice(p *[2][]byte, a, b []byte) {
	// amd64:`.*runtime[.]gcWriteBarrier4\(SB\)`
	// arm64:`.*runtime[.]gcWriteBarrier4\(SB\)`
	p[0] = a
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[1] = b
}

func combine4slice(p *[4][]byte, a, b, c, d []byte) {
	// amd64:`.*runtime[.]gcWriteBarrier8\(SB\)`
	// arm64:`.*runtime[.]gcWriteBarrier8\(SB\)`
	p[0] = a
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[1] = b
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[2] = c
	// amd64:-`.*runtime[.]gcWriteBarrier`
	// arm64:-`.*runtime[.]gcWriteBarrier`
	p[3] = d
}

func trickyWriteNil(p *int, q **int) {
	if p == nil {
		// We change "= p" to "= 0" in the prove pass, which
		// means we have one less pointer that needs to go
		// into the write barrier buffer.
		// amd64:`.*runtime[.]gcWriteBarrier1`
		*q = p
	}
}
