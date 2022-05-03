package build

// This file is being added for this cl only just to get builds to pass and
// let the other files be submitted unchanged.

const defaultCGO_ENABLED = ""

var cgoEnabled = map[string]bool{}

func getToolDir() string { return "" }
