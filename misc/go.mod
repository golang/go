// Module misc contains tests and binaries that pertain to specific build modes
// (cgo) and platforms (Android and iOS).
//
// The 'run' scripts in ../src execute these tests and binaries, which need to
// be in a module in order to build and run successfully in module mode.
// (Otherwise, they lack well-defined import paths, and module mode — unlike
// GOPATH mode — does not synthesize import paths from the absolute working
// directory.)
module misc

go 1.12
