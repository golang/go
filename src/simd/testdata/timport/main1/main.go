// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// main1 validates transitive SIMD rewriting across a package chain where
// neither main1 nor relay directly imports "simd":
//   - main imports relay.
//   - relay imports isimd and exports V = isimd.F() plus non-SIMD relayed vars.
//   - isimd imports simd, exports F() returning unexported type t (embedding
//     simd.Int8s), and initializes a non-SIMD dependency chain a -> b -> c -> d.
package main

import (
	"fmt"
	"os"
	"timport/relay"
)

// checkTransitiveEmbeddedSimdVar validates that calling a promoted SIMD method
// on relay.V (whose unexported type embeds simd.Int8s from timport/isimd)
// triggers midway rewriting in both relay and main1 and matches isimd.G().
func checkTransitiveEmbeddedSimdVar() {
	relayDotVdotReduceSum := relay.V.ReduceSum()
	fmt.Printf("relay.V.ReduceSum()=%d\n", relayDotVdotReduceSum)
	if relayDotVdotReduceSum != int8(relay.VdotReduceSum) {
		fmt.Printf("FAIL, %d != %d\n", relayDotVdotReduceSum, relay.VdotReduceSum)
		os.Exit(1)
	}
}

// checkRelayedInitChain validates that relay.PTwoP1 (derived in isimd via
// a -> b (SIMD) -> c -> d = 4*Int32s.Len()+1) initialized in dependency order.
func checkRelayedInitChain() {
	fmt.Printf("relay.PTwoP1=%d\n", relay.PTwoP1)
	if relay.PTwoP1&1 != 1 || relay.PTwoP1 < 17 {
		fmt.Printf("FAIL, expected relay.PTwoP1 = 2**N+1, got %d\n", relay.PTwoP1)
		os.Exit(1)
	}
}

func main() {
	checkTransitiveEmbeddedSimdVar()
	checkRelayedInitChain()
}
