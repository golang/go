// Export shim (M20-S1 violator): carries the //export so the main
// file's preamble can hold C definitions.
package main

import "C"

//export GoCallback
func GoCallback(v C.int) C.int { return v + 1 }
