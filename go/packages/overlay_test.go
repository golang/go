package packages_test

import (
	"log"
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
)

const commonMode = packages.NeedName | packages.NeedFiles |
	packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedSyntax

func TestOverlayChangesPackage(t *testing.T) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "fake",
		Files: map[string]interface{}{
			"a.go": "package foo\nfunc f(){}\n",
		},
		Overlay: map[string][]byte{
			"a.go": []byte("package foox\nfunc f(){}\n"),
		},
	}})
	defer exported.Cleanup()
	exported.Config.Mode = packages.NeedName

	initial, err := packages.Load(exported.Config,
		filepath.Dir(exported.File("fake", "a.go")))
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(initial) != 1 || initial[0].ID != "fake" || initial[0].Name != "foox" {
		t.Fatalf("got %v, expected [fake]", initial)
	}
	if len(initial[0].Errors) != 0 {
		t.Fatalf("got %v, expected no errors", initial[0].Errors)
	}
	log.SetFlags(0)
}
func TestOverlayChangesBothPackages(t *testing.T) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "fake",
		Files: map[string]interface{}{
			"a.go":      "package foo\nfunc g(){}\n",
			"a_test.go": "package foo\nfunc f(){}\n",
		},
		Overlay: map[string][]byte{
			"a.go":      []byte("package foox\nfunc g(){}\n"),
			"a_test.go": []byte("package foox\nfunc f(){}\n"),
		},
	}})
	defer exported.Cleanup()
	exported.Config.Mode = commonMode

	initial, err := packages.Load(exported.Config,
		filepath.Dir(exported.File("fake", "a.go")))
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(initial) != 3 {
		t.Errorf("got %d packges, expected 3", len(initial))
	}
	want := []struct {
		id, name string
		count    int
	}{
		{"fake", "foox", 1},
		{"fake [fake.test]", "foox", 2},
		{"fake.test", "main", 1},
	}
	for i := 0; i < 3; i++ {
		if ok := checkPkg(t, initial[i], want[i].id, want[i].name, want[i].count); !ok {
			t.Errorf("%d: got {%s %s %d}, expected %v", i, initial[i].ID,
				initial[i].Name, len(initial[i].Syntax), want[i])
		}
		if len(initial[i].Errors) != 0 {
			t.Errorf("%d: got %v, expected no errors", i, initial[i].Errors)
		}
	}
	log.SetFlags(0)
}

func TestOverlayChangesTestPackage(t *testing.T) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "fake",
		Files: map[string]interface{}{
			"a_test.go": "package foo\nfunc f(){}\n",
		},
		Overlay: map[string][]byte{
			"a_test.go": []byte("package foox\nfunc f(){}\n"),
		},
	}})
	defer exported.Cleanup()
	exported.Config.Mode = commonMode

	initial, err := packages.Load(exported.Config,
		filepath.Dir(exported.File("fake", "a_test.go")))
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(initial) != 3 {
		t.Errorf("got %d packges, expected 3", len(initial))
	}
	want := []struct {
		id, name string
		count    int
	}{
		{"fake", "foo", 0},
		{"fake [fake.test]", "foox", 1},
		{"fake.test", "main", 1},
	}
	for i := 0; i < 3; i++ {
		if ok := checkPkg(t, initial[i], want[i].id, want[i].name, want[i].count); !ok {
			t.Errorf("got {%s %s %d}, expected %v", initial[i].ID,
				initial[i].Name, len(initial[i].Syntax), want[i])
		}
	}
	if len(initial[0].Errors) != 0 {
		t.Fatalf("got %v, expected no errors", initial[0].Errors)
	}
	log.SetFlags(0)
}

func checkPkg(t *testing.T, p *packages.Package, id, name string, syntax int) bool {
	t.Helper()
	if p.ID == id && p.Name == name && len(p.Syntax) == syntax {
		return true
	}
	return false
}
