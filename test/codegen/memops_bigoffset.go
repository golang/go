// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type big1 struct {
	w [1<<30 - 1]uint32
}
type big2 struct {
	d [1<<29 - 1]uint64
}

func loadLargeOffset(sw *big1, sd *big2) (uint32, uint64) {

	// ppc64x:`MOVWZ\s+[0-9]+\(R[0-9]+\)`,-`ADD`
	a3 := sw.w[1<<10]
	// ppc64le/power10:`MOVWZ\s+[0-9]+\(R[0-9]+\),\sR[0-9]+`,-`ADD`
	// ppc64x/power9:`ADD`,`MOVWZ\s+\(R[0-9]+\),\sR[0-9]+`
	// ppc64x/power8:`ADD`,`MOVWZ\s+\(R[0-9]+\),\sR[0-9]+`
	b3 := sw.w[1<<16]
	// ppc64le/power10:`MOVWZ\s+[0-9]+\(R[0-9]+\),\sR[0-9]+`,-`ADD`
	// ppc64x/power9:`ADD`,`MOVWZ\s+\(R[0-9]+\),\sR[0-9]+`
	// ppc64x/power8:`ADD`,`MOVWZ\s+\(R[0-9]+\),\sR[0-9]+`
	c3 := sw.w[1<<28]
	// ppc64x:`MOVWZ\s+\(R[0-9]+\)\(R[0-9]+\),\sR[0-9]+`
	d3 := sw.w[1<<29]
	// ppc64x:`MOVD\s+[0-9]+\(R[0-9]+\)`,-`ADD`
	a4 := sd.d[1<<10]
	// ppc64le/power10:`MOVD\s+[0-9]+\(R[0-9]+\)`,-`ADD`
	// ppc64x/power9:`ADD`,`MOVD\s+\(R[0-9]+\),\sR[0-9]+`
	// ppc64x/power8:`ADD`,`MOVD\s+\(R[0-9]+\),\sR[0-9]+`
	b4 := sd.d[1<<16]
	// ppc64le/power10`:`MOVD\s+[0-9]+\(R[0-9]+\)`,-`ADD`
	// ppc64x/power9:`ADD`,`MOVD\s+\(R[0-9]+\),\sR[0-9]+`
	// ppc64x/power8:`ADD`,`MOVD\s+\(R[0-9]+\),\sR[0-9]+`
	c4 := sd.d[1<<27]
	// ppc64x:`MOVD\s+\(R[0-9]+\)\(R[0-9]+\),\sR[0-9]+`
	d4 := sd.d[1<<28]

	return a3 + b3 + c3 + d3, a4 + b4 + c4 + d4
}

func storeLargeOffset(sw *big1, sd *big2) {
	// ppc64x:`MOVW\s+R[0-9]+,\s[0-9]+\(R[0-9]+\)`,-`ADD`
	sw.w[1<<10] = uint32(10)
	// ppc64le/power10:`MOVW\s+R[0-9]+,\s[0-9]+\(R[0-9]+\)`,-`ADD`
	// ppc64x/power9:`MOVW\s+R[0-9]+\,\s\(R[0-9]+\)`,`ADD`
	// ppc64x/power8:`MOVW\s+R[0-9]+\,\s\(R[0-9]+\)`,`ADD`
	sw.w[1<<16] = uint32(20)
	// ppc64le/power10:`MOVW\s+R[0-9]+,\s[0-9]+\(R[0-9]+\)`,-`ADD`
	// ppc64x/power9:`MOVW\s+R[0-9]+,\s\(R[0-9]+\)`,`ADD`
	// ppc64x/power8:`MOVW\s+R[0-9]+,\s\(R[0-9]+\)`,`ADD`
	sw.w[1<<28] = uint32(30)
	// ppc64x:`MOVW\s+R[0-9]+,\s\(R[0-9]+\)`
	sw.w[1<<29] = uint32(40)
	// ppc64x:`MOVD\s+R[0-9]+,\s[0-9]+\(R[0-9]+\)`,-`ADD`
	sd.d[1<<10] = uint64(40)
	// ppc64le/power10:`MOVD\s+R[0-9]+,\s[0-9]+\(R[0-9]+\)`,-`ADD`
	// ppc64x/power9:`MOVD\s+R[0-9]+,\s\(R[0-9]+\)`,`ADD`
	// ppc64x/power8:`MOVD\s+R[0-9]+,\s\(R[0-9]+\)`,`ADD`
	sd.d[1<<16] = uint64(50)
	// ppc64le/power10`:`MOVD\s+R[0-9]+,\s[0-9]+\(R[0-9]+\)`,-`ADD`
	// ppc64x/power9:`MOVD\s+R[0-9]+,\s\(R[0-9]+\)`,`ADD`
	// ppc64x/power8:`MOVD\s+R[0-9]+,\s\(R[0-9]+\)`,`ADD`
	sd.d[1<<27] = uint64(60)
	// ppc64x:`MOVD\s+R[0-9]+,\s\(R[0-9]+\)`
	sd.d[1<<28] = uint64(70)
}
