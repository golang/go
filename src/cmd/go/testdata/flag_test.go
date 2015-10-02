package flag_test

import (
	"flag"
	"log"
	"testing"
)

var v = flag.Int("v", 0, "v flag")

// Run this as go test pkg -v=7
func TestVFlagIsSet(t *testing.T) {
	if *v != 7 {
		log.Fatal("v flag not set")
	}
}
