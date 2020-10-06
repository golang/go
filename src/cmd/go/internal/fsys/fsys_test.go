package fsys

import (
	"cmd/go/internal/txtar"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// initOverlay resets the overlay state to reflect the config.
// config should be a text archive string. The comment is the overlay config
// json, and the files, in the archive are laid out in a temp directory
// that cwd is set to.
func initOverlay(t *testing.T, config string) {
	t.Helper()

	// Create a temporary directory and chdir to it.
	prevwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cwd = filepath.Join(t.TempDir(), "root")
	if err := os.Mkdir(cwd, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(cwd); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		overlay = nil
		if err := os.Chdir(prevwd); err != nil {
			t.Fatal(err)
		}
	})

	a := txtar.Parse([]byte(config))
	for _, f := range a.Files {
		name := filepath.Join(cwd, f.Name)
		if err := os.MkdirAll(filepath.Dir(name), 0777); err != nil {
			t.Fatal(err)
		}
		if err := ioutil.WriteFile(name, f.Data, 0666); err != nil {
			t.Fatal(err)
		}
	}

	var overlayJSON OverlayJSON
	if err := json.Unmarshal(a.Comment, &overlayJSON); err != nil {
		t.Fatal(fmt.Errorf("parsing overlay JSON: %v", err))
	}

	initFromJSON(overlayJSON)
}

func TestIsDir(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"subdir2/file2.txt":  "overlayfiles/subdir2_file2.txt",
		"subdir4":            "overlayfiles/subdir4",
		"subdir3/file3b.txt": "overlayfiles/subdir3_file3b.txt",
		"subdir5":            "",
		"subdir6":            ""
	}
}
-- subdir1/file1.txt --

-- subdir3/file3a.txt --
33
-- subdir4/file4.txt --
444
-- overlayfiles/subdir2_file2.txt --
2
-- overlayfiles/subdir3_file3b.txt --
66666
-- overlayfiles/subdir4 --
x
-- subdir6/file6.txt --
six
`)

	testCases := []struct {
		path          string
		want, wantErr bool
	}{
		{"", true, true},
		{".", true, false},
		{cwd, true, false},
		{cwd + string(filepath.Separator), true, false},
		// subdir1 is only on disk
		{filepath.Join(cwd, "subdir1"), true, false},
		{"subdir1", true, false},
		{"subdir1" + string(filepath.Separator), true, false},
		{"subdir1/file1.txt", false, false},
		{"subdir1/doesntexist.txt", false, true},
		{"doesntexist", false, true},
		// subdir2 is only in overlay
		{filepath.Join(cwd, "subdir2"), true, false},
		{"subdir2", true, false},
		{"subdir2" + string(filepath.Separator), true, false},
		{"subdir2/file2.txt", false, false},
		{"subdir2/doesntexist.txt", false, true},
		// subdir3 has files on disk and in overlay
		{filepath.Join(cwd, "subdir3"), true, false},
		{"subdir3", true, false},
		{"subdir3" + string(filepath.Separator), true, false},
		{"subdir3/file3a.txt", false, false},
		{"subdir3/file3b.txt", false, false},
		{"subdir3/doesntexist.txt", false, true},
		// subdir4 is overlaid with a file
		{filepath.Join(cwd, "subdir4"), false, false},
		{"subdir4", false, false},
		{"subdir4" + string(filepath.Separator), false, false},
		{"subdir4/file4.txt", false, false},
		{"subdir4/doesntexist.txt", false, false},
		// subdir5 doesn't exist, and is overlaid with a "delete" entry
		{filepath.Join(cwd, "subdir5"), false, false},
		{"subdir5", false, false},
		{"subdir5" + string(filepath.Separator), false, false},
		{"subdir5/file5.txt", false, false},
		{"subdir5/doesntexist.txt", false, false},
		// subdir6 does exist, and is overlaid with a "delete" entry
		{filepath.Join(cwd, "subdir6"), false, false},
		{"subdir6", false, false},
		{"subdir6" + string(filepath.Separator), false, false},
		{"subdir6/file6.txt", false, false},
		{"subdir6/doesntexist.txt", false, false},
	}

	for _, tc := range testCases {
		got, err := IsDir(tc.path)
		if err != nil {
			if !tc.wantErr {
				t.Errorf("IsDir(%q): got error with string %q, want no error", tc.path, err.Error())
			}
			continue
		}
		if tc.wantErr {
			t.Errorf("IsDir(%q): got no error, want error", tc.path)
		}
		if tc.want != got {
			t.Errorf("IsDir(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

func TestReadDir(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"subdir2/file2.txt":                 "overlayfiles/subdir2_file2.txt",
		"subdir4":                           "overlayfiles/subdir4",
		"subdir3/file3b.txt":                "overlayfiles/subdir3_file3b.txt",
		"subdir5":                           "",
		"subdir6/asubsubdir/afile.txt":      "overlayfiles/subdir6_asubsubdir_afile.txt",
		"subdir6/asubsubdir/zfile.txt":      "overlayfiles/subdir6_asubsubdir_zfile.txt",
		"subdir6/zsubsubdir/file.txt":       "overlayfiles/subdir6_zsubsubdir_file.txt",
		"subdir7/asubsubdir/file.txt":       "overlayfiles/subdir7_asubsubdir_file.txt",
		"subdir7/zsubsubdir/file.txt":       "overlayfiles/subdir7_zsubsubdir_file.txt",
		"subdir8/doesntexist":               "this_file_doesnt_exist_anywhere",
		"other/pointstodir":                 "overlayfiles/this_is_a_directory",
		"parentoverwritten/subdir1":         "overlayfiles/parentoverwritten_subdir1",
		"subdir9/this_file_is_overlaid.txt": "overlayfiles/subdir9_this_file_is_overlaid.txt",
		"subdir10/only_deleted_file.txt":    "",
		"subdir11/deleted.txt":              "",
		"subdir11":                          "overlayfiles/subdir11",
		"textfile.txt/file.go":              "overlayfiles/textfile_txt_file.go"
	}
}
-- subdir1/file1.txt --

-- subdir3/file3a.txt --
33
-- subdir4/file4.txt --
444
-- subdir6/file.txt --
-- subdir6/asubsubdir/file.txt --
-- subdir6/anothersubsubdir/file.txt --
-- subdir9/this_file_is_overlaid.txt --
-- subdir10/only_deleted_file.txt --
this will be deleted in overlay
-- subdir11/deleted.txt --
-- parentoverwritten/subdir1/subdir2/subdir3/file.txt --
-- textfile.txt --
this will be overridden by textfile.txt/file.go
-- overlayfiles/subdir2_file2.txt --
2
-- overlayfiles/subdir3_file3b.txt --
66666
-- overlayfiles/subdir4 --
x
-- overlayfiles/subdir6_asubsubdir_afile.txt --
-- overlayfiles/subdir6_asubsubdir_zfile.txt --
-- overlayfiles/subdir6_zsubsubdir_file.txt --
-- overlayfiles/subdir7_asubsubdir_file.txt --
-- overlayfiles/subdir7_zsubsubdir_file.txt --
-- overlayfiles/parentoverwritten_subdir1 --
x
-- overlayfiles/subdir9_this_file_is_overlaid.txt --
99999999
-- overlayfiles/subdir11 --
-- overlayfiles/this_is_a_directory/file.txt --
-- overlayfiles/textfile_txt_file.go --
x
`)

	testCases := map[string][]struct {
		name  string
		size  int64
		isDir bool
	}{
		".": {
			{"other", 0, true},
			{"overlayfiles", 0, true},
			{"parentoverwritten", 0, true},
			{"subdir1", 0, true},
			{"subdir10", 0, true},
			{"subdir11", 0, false},
			{"subdir2", 0, true},
			{"subdir3", 0, true},
			{"subdir4", 2, false},
			// no subdir5.
			{"subdir6", 0, true},
			{"subdir7", 0, true},
			{"subdir8", 0, true},
			{"subdir9", 0, true},
			{"textfile.txt", 0, true},
		},
		"subdir1": {{"file1.txt", 1, false}},
		"subdir2": {{"file2.txt", 2, false}},
		"subdir3": {{"file3a.txt", 3, false}, {"file3b.txt", 6, false}},
		"subdir6": {
			{"anothersubsubdir", 0, true},
			{"asubsubdir", 0, true},
			{"file.txt", 0, false},
			{"zsubsubdir", 0, true},
		},
		"subdir6/asubsubdir": {{"afile.txt", 0, false}, {"file.txt", 0, false}, {"zfile.txt", 0, false}},
		"subdir8":            {{"doesntexist", 0, false}}, // entry is returned even if destination file doesn't exist
		// check that read dir actually redirects files that already exist
		// the original this_file_is_overlaid.txt is empty
		"subdir9":           {{"this_file_is_overlaid.txt", 9, false}},
		"subdir10":          {},
		"parentoverwritten": {{"subdir1", 2, false}},
		"textfile.txt":      {{"file.go", 2, false}},
	}

	for dir, want := range testCases {
		fis, err := ReadDir(dir)
		if err != nil {
			t.Fatalf("ReadDir(%q): got error %q, want no error", dir, err)
		}
		if len(fis) != len(want) {
			t.Fatalf("ReadDir(%q) result: got %v entries; want %v entries", dir, len(fis), len(want))
		}
		for i := range fis {
			if fis[i].Name() != want[i].name {
				t.Fatalf("ReadDir(%q) entry %v: got Name() = %v, want %v", dir, i, fis[i].Name(), want[i].name)
			}
			if fis[i].IsDir() != want[i].isDir {
				t.Fatalf("ReadDir(%q) entry %v: got IsDir() = %v, want %v", dir, i, fis[i].IsDir(), want[i].isDir)
			}
			if want[i].isDir {
				// We don't try to get size right for directories
				continue
			}
			if fis[i].Size() != want[i].size {
				t.Fatalf("ReadDir(%q) entry %v: got Size() = %v, want %v", dir, i, fis[i].Size(), want[i].size)
			}
		}
	}

	errCases := []string{
		"subdir1/file1.txt", // regular file on disk
		"subdir2/file2.txt", // regular file in overlay
		"subdir4",           // directory overlaid with regular file
		"subdir5",           // directory deleted in overlay
		"parentoverwritten/subdir1/subdir2/subdir3", // parentoverwritten/subdir1 overlaid with regular file
		"parentoverwritten/subdir1/subdir2",         // parentoverwritten/subdir1 overlaid with regular file
		"subdir11",                                  // directory with deleted child, overlaid with regular file
		"other/pointstodir",
	}

	for _, dir := range errCases {
		_, gotErr := ReadDir(dir)
		if gotErr == nil {
			t.Errorf("ReadDir(%q): got no error, want error", dir)
		} else if _, ok := gotErr.(*os.PathError); !ok {
			t.Errorf("ReadDir(%q): got error with string %q and type %T, want os.PathError", dir, gotErr.Error(), gotErr)
		}
	}
}

func TestOverlayPath(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"subdir2/file2.txt":                 "overlayfiles/subdir2_file2.txt",
		"subdir3/doesntexist":               "this_file_doesnt_exist_anywhere",
		"subdir4/this_file_is_overlaid.txt": "overlayfiles/subdir4_this_file_is_overlaid.txt",
		"subdir5/deleted.txt":               "",
		"parentoverwritten/subdir1":         ""
	}
}
-- subdir1/file1.txt --
file 1
-- subdir4/this_file_is_overlaid.txt --
these contents are replaced by the overlay
-- parentoverwritten/subdir1/subdir2/subdir3/file.txt --
-- subdir5/deleted.txt --
deleted
-- overlayfiles/subdir2_file2.txt --
file 2
-- overlayfiles/subdir4_this_file_is_overlaid.txt --
99999999
`)

	testCases := []struct {
		path     string
		wantPath string
		wantOK   bool
	}{
		{"subdir1/file1.txt", "subdir1/file1.txt", false},
		// OverlayPath returns false for directories
		{"subdir2", "subdir2", false},
		{"subdir2/file2.txt", filepath.Join(cwd, "overlayfiles/subdir2_file2.txt"), true},
		// OverlayPath doesn't stat a file to see if it exists, so it happily returns
		// the 'to' path and true even if the 'to' path doesn't exist on disk.
		{"subdir3/doesntexist", filepath.Join(cwd, "this_file_doesnt_exist_anywhere"), true},
		// Like the subdir2/file2.txt case above, but subdir4 exists on disk, but subdir2 does not.
		{"subdir4/this_file_is_overlaid.txt", filepath.Join(cwd, "overlayfiles/subdir4_this_file_is_overlaid.txt"), true},
		{"subdir5", "subdir5", false},
		{"subdir5/deleted.txt", "", true},
	}

	for _, tc := range testCases {
		gotPath, gotOK := OverlayPath(tc.path)
		if gotPath != tc.wantPath || gotOK != tc.wantOK {
			t.Errorf("OverlayPath(%q): got %v, %v; want %v, %v",
				tc.path, gotPath, gotOK, tc.wantPath, tc.wantOK)
		}
	}
}

func TestOpen(t *testing.T) {
	initOverlay(t, `
{
    "Replace": {
		"subdir2/file2.txt":                  "overlayfiles/subdir2_file2.txt",
		"subdir3/doesntexist":                "this_file_doesnt_exist_anywhere",
		"subdir4/this_file_is_overlaid.txt":  "overlayfiles/subdir4_this_file_is_overlaid.txt",
		"subdir5/deleted.txt":                "",
		"parentoverwritten/subdir1":          "",
		"childoverlay/subdir1.txt/child.txt": "overlayfiles/child.txt",
		"subdir11/deleted.txt":               "",
		"subdir11":                           "overlayfiles/subdir11",
		"parentdeleted":                      "",
		"parentdeleted/file.txt":             "overlayfiles/parentdeleted_file.txt"
	}
}
-- subdir11/deleted.txt --
-- subdir1/file1.txt --
file 1
-- subdir4/this_file_is_overlaid.txt --
these contents are replaced by the overlay
-- parentoverwritten/subdir1/subdir2/subdir3/file.txt --
-- childoverlay/subdir1.txt --
this file doesn't exist because the path
childoverlay/subdir1.txt/child.txt is in the overlay
-- subdir5/deleted.txt --
deleted
-- parentdeleted --
this will be deleted so that parentdeleted/file.txt can exist
-- overlayfiles/subdir2_file2.txt --
file 2
-- overlayfiles/subdir4_this_file_is_overlaid.txt --
99999999
-- overlayfiles/child.txt --
-- overlayfiles/subdir11 --
11
-- overlayfiles/parentdeleted_file.txt --
this can exist because the parent directory is deleted
`)

	testCases := []struct {
		path         string
		wantContents string
		isErr        bool
	}{
		{"subdir1/file1.txt", "file 1\n", false},
		{"subdir2/file2.txt", "file 2\n", false},
		{"subdir3/doesntexist", "", true},
		{"subdir4/this_file_is_overlaid.txt", "99999999\n", false},
		{"subdir5/deleted.txt", "", true},
		{"parentoverwritten/subdir1/subdir2/subdir3/file.txt", "", true},
		{"childoverlay/subdir1.txt", "", true},
		{"subdir11", "11\n", false},
		{"parentdeleted/file.txt", "this can exist because the parent directory is deleted\n", false},
	}

	for _, tc := range testCases {
		f, err := Open(tc.path)
		if tc.isErr {
			if err == nil {
				f.Close()
				t.Errorf("Open(%q): got no error, but want error", tc.path)
			}
			continue
		}
		if err != nil {
			t.Errorf("Open(%q): got error %v, want nil", tc.path, err)
			continue
		}
		contents, err := ioutil.ReadAll(f)
		if err != nil {
			t.Errorf("unexpected error reading contents of file: %v", err)
		}
		if string(contents) != tc.wantContents {
			t.Errorf("contents of file opened with Open(%q): got %q, want %q",
				tc.path, contents, tc.wantContents)
		}
		f.Close()
	}
}

func TestIsDirWithGoFiles(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"goinoverlay/file.go":       "dummy",
		"directory/removed/by/file": "dummy",
		"directory_with_go_dir/dir.go/file.txt": "dummy",
		"otherdirectory/deleted.go": "",
		"nonexistentdirectory/deleted.go": "",
		"textfile.txt/file.go": "dummy"
	}
}
-- dummy --
a destination file for the overlay entries to point to
contents don't matter for this test
-- nogo/file.txt --
-- goondisk/file.go --
-- goinoverlay/file.txt --
-- directory/removed/by/file/in/overlay/file.go --
-- otherdirectory/deleted.go --
-- textfile.txt --
`)

	testCases := []struct {
		dir     string
		want    bool
		wantErr bool
	}{
		{"nogo", false, false},
		{"goondisk", true, false},
		{"goinoverlay", true, false},
		{"directory/removed/by/file/in/overlay", false, false},
		{"directory_with_go_dir", false, false},
		{"otherdirectory", false, false},
		{"nonexistentdirectory", false, false},
		{"textfile.txt", true, false},
	}

	for _, tc := range testCases {
		got, gotErr := IsDirWithGoFiles(tc.dir)
		if tc.wantErr {
			if gotErr == nil {
				t.Errorf("IsDirWithGoFiles(%q): got %v, %v; want non-nil error", tc.dir, got, gotErr)
			}
			continue
		}
		if gotErr != nil {
			t.Errorf("IsDirWithGoFiles(%q): got %v, %v; want nil error", tc.dir, got, gotErr)
		}
		if got != tc.want {
			t.Errorf("IsDirWithGoFiles(%q) = %v; want %v", tc.dir, got, tc.want)
		}
	}
}

func TestWalk(t *testing.T) {
	type file struct {
		path  string
		name  string
		size  int64
		mode  os.FileMode
		isDir bool
	}
	testCases := []struct {
		name      string
		overlay   string
		root      string
		wantFiles []file
	}{
		{"no overlay", `
{}
-- file.txt --
`,
			".",
			[]file{
				{".", "root", 0, os.ModeDir | 0700, true},
				{"file.txt", "file.txt", 0, 0600, false},
			},
		},
		{"overlay with different file", `
{
	"Replace": {
		"file.txt": "other.txt"
	}
}
-- file.txt --
-- other.txt --
contents of other file
`,
			".",
			[]file{
				{".", "root", 0, os.ModeDir | 0500, true},
				{"file.txt", "file.txt", 23, 0600, false},
				{"other.txt", "other.txt", 23, 0600, false},
			},
		},
		{"overlay with new file", `
{
	"Replace": {
		"file.txt": "other.txt"
	}
}
-- other.txt --
contents of other file
`,
			".",
			[]file{
				{".", "root", 0, os.ModeDir | 0500, true},
				{"file.txt", "file.txt", 23, 0600, false},
				{"other.txt", "other.txt", 23, 0600, false},
			},
		},
		{"overlay with new directory", `
{
	"Replace": {
		"dir/file.txt": "other.txt"
	}
}
-- other.txt --
contents of other file
`,
			".",
			[]file{
				{".", "root", 0, os.ModeDir | 0500, true},
				{"dir", "dir", 0, os.ModeDir | 0500, true},
				{"dir" + string(filepath.Separator) + "file.txt", "file.txt", 23, 0600, false},
				{"other.txt", "other.txt", 23, 0600, false},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			initOverlay(t, tc.overlay)

			var got []file
			Walk(tc.root, func(path string, info os.FileInfo, err error) error {
				got = append(got, file{path, info.Name(), info.Size(), info.Mode(), info.IsDir()})
				return nil
			})

			if len(got) != len(tc.wantFiles) {
				t.Errorf("Walk: saw %#v in walk; want %#v", got, tc.wantFiles)
			}
			for i := 0; i < len(got) && i < len(tc.wantFiles); i++ {
				if got[i].path != tc.wantFiles[i].path {
					t.Errorf("path of file #%v in walk, got %q, want %q", i, got[i].path, tc.wantFiles[i].path)
				}
				if got[i].name != tc.wantFiles[i].name {
					t.Errorf("name of file #%v in walk, got %q, want %q", i, got[i].name, tc.wantFiles[i].name)
				}
				if got[i].mode&(os.ModeDir|0700) != tc.wantFiles[i].mode {
					t.Errorf("mode&(os.ModeDir|0700) for mode of file #%v in walk, got %v, want %v", i, got[i].mode&(os.ModeDir|0700), tc.wantFiles[i].mode)
				}
				if got[i].isDir != tc.wantFiles[i].isDir {
					t.Errorf("isDir for file #%v in walk, got %v, want %v", i, got[i].isDir, tc.wantFiles[i].isDir)
				}
				if tc.wantFiles[i].isDir {
					continue // don't check size for directories
				}
				if got[i].size != tc.wantFiles[i].size {
					t.Errorf("size of file #%v in walk, got %v, want %v", i, got[i].size, tc.wantFiles[i].size)
				}
			}
		})
	}
}

func TestWalk_SkipDir(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"skipthisdir/file.go": "dummy.txt",
		"dontskip/file.go": "dummy.txt",
		"dontskip/skip/file.go": "dummy.txt"
	}
}
-- dummy.txt --
`)

	var seen []string
	Walk(".", func(path string, info os.FileInfo, err error) error {
		seen = append(seen, path)
		if path == "skipthisdir" || path == filepath.Join("dontskip", "skip") {
			return filepath.SkipDir
		}
		return nil
	})

	wantSeen := []string{".", "dontskip", filepath.Join("dontskip", "file.go"), filepath.Join("dontskip", "skip"), "dummy.txt", "skipthisdir"}

	if len(seen) != len(wantSeen) {
		t.Errorf("paths seen in walk: got %v entries; want %v entries", len(seen), len(wantSeen))
	}

	for i := 0; i < len(seen) && i < len(wantSeen); i++ {
		if seen[i] != wantSeen[i] {
			t.Errorf("path #%v seen walking tree: want %q, got %q", i, seen[i], wantSeen[i])
		}
	}
}

func TestWalk_Error(t *testing.T) {
	initOverlay(t, "{}")

	alreadyCalled := false
	err := Walk("foo", func(path string, info os.FileInfo, err error) error {
		if alreadyCalled {
			t.Fatal("expected walk function to be called exactly once, but it was called more than once")
		}
		alreadyCalled = true
		return errors.New("returned from function")
	})
	if !alreadyCalled {
		t.Fatal("expected walk function to be called exactly once, but it was never called")

	}
	if err == nil {
		t.Fatalf("Walk: got no error, want error")
	}
	if err.Error() != "returned from function" {
		t.Fatalf("Walk: got error %v, want \"returned from function\" error", err)
	}
}

func TestWalk_Symlink(t *testing.T) {
	initOverlay(t, `{
	"Replace": {"overlay_symlink": "symlink"}
}
-- dir/file --`)

	// Create symlink
	if err := os.Symlink("dir", "symlink"); err != nil {
		t.Error(err)
	}

	testCases := []struct {
		name      string
		dir       string
		wantFiles []string
	}{
		{"control", "dir", []string{"dir", "dir" + string(filepath.Separator) + "file"}},
		// ensure Walk doesn't wolk into the directory pointed to by the symlink
		// (because it's supposed to use Lstat instead of Stat.
		{"symlink_to_dir", "symlink", []string{"symlink"}},
		{"overlay_to_symlink_to_dir", "overlay_symlink", []string{"overlay_symlink"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var got []string

			err := Walk(tc.dir, func(path string, info os.FileInfo, err error) error {
				got = append(got, path)
				if err != nil {
					t.Errorf("walkfn: got non nil err argument: %v, want nil err argument", err)
				}
				return nil
			})
			if err != nil {
				t.Errorf("Walk: got error %q, want nil", err)
			}

			if !reflect.DeepEqual(got, tc.wantFiles) {
				t.Errorf("files examined by walk: got %v, want %v", got, tc.wantFiles)
			}
		})
	}

}

func TestLstat(t *testing.T) {
	type file struct {
		name  string
		size  int64
		mode  os.FileMode // mode & (os.ModeDir|0x700): only check 'user' permissions
		isDir bool
	}

	testCases := []struct {
		name    string
		overlay string
		path    string

		want    file
		wantErr bool
	}{
		{
			"regular_file",
			`{}
-- file.txt --
contents`,
			"file.txt",
			file{"file.txt", 9, 0600, false},
			false,
		},
		{
			"new_file_in_overlay",
			`{"Replace": {"file.txt": "dummy.txt"}}
-- dummy.txt --
contents`,
			"file.txt",
			file{"file.txt", 9, 0600, false},
			false,
		},
		{
			"file_replaced_in_overlay",
			`{"Replace": {"file.txt": "dummy.txt"}}
-- file.txt --
-- dummy.txt --
contents`,
			"file.txt",
			file{"file.txt", 9, 0600, false},
			false,
		},
		{
			"file_cant_exist",
			`{"Replace": {"deleted": "dummy.txt"}}
-- deleted/file.txt --
-- dummy.txt --
`,
			"deleted/file.txt",
			file{},
			true,
		},
		{
			"deleted",
			`{"Replace": {"deleted": ""}}
-- deleted --
`,
			"deleted",
			file{},
			true,
		},
		{
			"dir_on_disk",
			`{}
-- dir/foo.txt --
`,
			"dir",
			file{"dir", 0, 0700 | os.ModeDir, true},
			false,
		},
		{
			"dir_in_overlay",
			`{"Replace": {"dir/file.txt": "dummy.txt"}}
-- dummy.txt --
`,
			"dir",
			file{"dir", 0, 0500 | os.ModeDir, true},
			false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			initOverlay(t, tc.overlay)
			got, err := lstat(tc.path)
			if tc.wantErr {
				if err == nil {
					t.Errorf("lstat(%q): got no error, want error", tc.path)
				}
				return
			}
			if err != nil {
				t.Fatalf("lstat(%q): got error %v, want no error", tc.path, err)
			}
			if got.Name() != tc.want.name {
				t.Errorf("lstat(%q).Name(): got %q, want %q", tc.path, got.Name(), tc.want.name)
			}
			if got.Mode()&(os.ModeDir|0700) != tc.want.mode {
				t.Errorf("lstat(%q).Mode()&(os.ModeDir|0700): got %v, want %v", tc.path, got.Mode()&(os.ModeDir|0700), tc.want.mode)
			}
			if got.IsDir() != tc.want.isDir {
				t.Errorf("lstat(%q).IsDir(): got %v, want %v", tc.path, got.IsDir(), tc.want.isDir)
			}
			if tc.want.isDir {
				return // don't check size for directories
			}
			if got.Size() != tc.want.size {
				t.Errorf("lstat(%q).Size(): got %v, want %v", tc.path, got.Size(), tc.want.size)
			}
		})
	}
}
