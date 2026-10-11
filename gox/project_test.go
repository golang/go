package gox

import (
	"go/build"
	"os"
	"path/filepath"
	"testing"
)

func TestOverlaySelectionAndCache(t *testing.T) {
	d := t.TempDir()
	cache := t.TempDir()
	os.WriteFile(filepath.Join(d, "main.gox"), []byte("package p;var x=F\"{2}\""), 0600)
	os.WriteFile(filepath.Join(d, "bad_windows.gox"), []byte("not valid"), 0600)
	ctx := build.Default
	ctx.GOOS = "linux"
	opts := ProjectOptions{Options: Options{BareJSX: true}, Context: ctx, CacheDir: cache}
	o, e := PrepareOverlay(d, opts)
	if e != nil {
		t.Fatal(e)
	}
	if len(o.Replace) != 1 {
		t.Fatal(o)
	}
	path := o.Replace[filepath.Join(d, "main.go")]
	b, e := os.ReadFile(path)
	if e != nil || len(b) == 0 {
		t.Fatal(e)
	}
	o2, e := PrepareOverlay(d, opts)
	if e != nil || o2.Replace[filepath.Join(d, "main.go")] != path {
		t.Fatal("unstable cache", e)
	}
	os.WriteFile(filepath.Join(d, "main.go"), b, 0600)
	if _, e = PrepareOverlay(d, opts); e == nil {
		t.Fatal("accepted output collision")
	}
}
