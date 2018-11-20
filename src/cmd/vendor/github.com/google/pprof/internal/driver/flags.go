package driver

import (
	"flag"
	"strings"
)

// GoFlags implements the plugin.FlagSet interface.
type GoFlags struct {
	UsageMsgs []string
}

// Bool implements the plugin.FlagSet interface.
func (*GoFlags) Bool(o string, d bool, c string) *bool {
	return flag.Bool(o, d, c)
}

// Int implements the plugin.FlagSet interface.
func (*GoFlags) Int(o string, d int, c string) *int {
	return flag.Int(o, d, c)
}

// Float64 implements the plugin.FlagSet interface.
func (*GoFlags) Float64(o string, d float64, c string) *float64 {
	return flag.Float64(o, d, c)
}

// String implements the plugin.FlagSet interface.
func (*GoFlags) String(o, d, c string) *string {
	return flag.String(o, d, c)
}

// BoolVar implements the plugin.FlagSet interface.
func (*GoFlags) BoolVar(b *bool, o string, d bool, c string) {
	flag.BoolVar(b, o, d, c)
}

// IntVar implements the plugin.FlagSet interface.
func (*GoFlags) IntVar(i *int, o string, d int, c string) {
	flag.IntVar(i, o, d, c)
}

// Float64Var implements the plugin.FlagSet interface.
// the value of the flag.
func (*GoFlags) Float64Var(f *float64, o string, d float64, c string) {
	flag.Float64Var(f, o, d, c)
}

// StringVar implements the plugin.FlagSet interface.
func (*GoFlags) StringVar(s *string, o, d, c string) {
	flag.StringVar(s, o, d, c)
}

// StringList implements the plugin.FlagSet interface.
func (*GoFlags) StringList(o, d, c string) *[]*string {
	return &[]*string{flag.String(o, d, c)}
}

// ExtraUsage implements the plugin.FlagSet interface.
func (f *GoFlags) ExtraUsage() string {
	return strings.Join(f.UsageMsgs, "\n")
}

// AddExtraUsage implements the plugin.FlagSet interface.
func (f *GoFlags) AddExtraUsage(eu string) {
	f.UsageMsgs = append(f.UsageMsgs, eu)
}

// Parse implements the plugin.FlagSet interface.
func (*GoFlags) Parse(usage func()) []string {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		usage()
	}
	return args
}
