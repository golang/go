package buildutil_test

import (
	"flag"
	"go/build"
	"reflect"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func TestTags(t *testing.T) {
	f := flag.NewFlagSet("TestTags", flag.PanicOnError)
	var ctxt build.Context
	f.Var((*buildutil.TagsFlag)(&ctxt.BuildTags), "tags", buildutil.TagsFlagDoc)
	f.Parse([]string{"-tags", ` 'one'"two"	'three "four"'`, "rest"})

	// BuildTags
	want := []string{"one", "two", "three \"four\""}
	if !reflect.DeepEqual(ctxt.BuildTags, want) {
		t.Errorf("BuildTags = %q, want %q", ctxt.BuildTags, want)
	}

	// Args()
	if want := []string{"rest"}; !reflect.DeepEqual(f.Args(), want) {
		t.Errorf("f.Args() = %q, want %q", f.Args(), want)
	}
}
